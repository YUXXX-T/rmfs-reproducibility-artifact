# Simulator source guide

This page connects the simulator description to the exact implementation under
`src/`. It is a reading guide for the evaluated code path: it identifies where
the world is constructed, which component owns each transition, and where the
paper's evaluation adds metrics around the common engine.

## Which entry point is used?

There are three related entry points, but they serve different purposes.

| Use | Entry point | Role |
|---|---|---|
| Direct simulator or smoke run | `src/main.py` | Loads one JSON configuration, seeds Python/NumPy, constructs the configured policies, and runs the engine. |
| Paper evaluation | `src/WorldModel/evaluation/evaluate_online_v6.py`, function `_run_one_assigner` | Sets the evaluation seed, resets global order/task IDs, applies the requested tick limit and arrival manifest, attaches paper metric probes, and runs the same engine. |
| Optional RL-style API | `src/Env/rmfs_env.py`, class `RMFSEnv` | Wraps the engine as a PettingZoo parallel environment. It is not the execution path used for the reported PP campaigns. |

The paper runner delegates engine construction to `_build_engine` in
`src/WorldModel/evaluation/evaluate.py`. Consequently, Greedy, the dispatch
baselines, and the proposed method share the same world state, station logic,
movement code, and path planner. They differ at the configured task-assignment
policy, not at the simulator transition function.

## Source-tree map

| Path | Responsibility |
|---|---|
| `src/Config/config_loader.py` | Parses simulator JSON into typed configuration objects. |
| `src/Engine/simulation_engine.py` | Owns the run loop and the ordered transition for one tick. |
| `src/WorldState/` | Holds the mutable source of truth for the map, robots, pods, stations, orders, and tasks. |
| `src/Policies/OrderGenerator/` | Creates stochastic orders or replays a fixed arrival manifest. |
| `src/Policies/TaskAssigner/` | Creates assignment contexts, selects robots, and commits task chains. |
| `src/Policies/PathPlanner/` | Plans robot motion; the main result uses prioritized planning (PP). |
| `src/Policies/PodInitializer/` | Places pods and initializes SKU inventories. |
| `src/Policies/PodRetriever/` | Selects the pod or pods that cover an order's unfulfilled demand. |
| `src/Policies/PodReturnPlanner/` | Selects the post-service return location. |
| `src/Metrics/tracker.py` | Records basic per-tick engine counters and assignment/planning wall time. |
| `src/WorldModel/evaluation/` | Builds paper runs, replays paired arrivals, and collects the extended evaluation metrics. |
| `src/Visualization/` | Optional renderers; no renderer is active in the reported batch runs. |

`src/rmfs_artifact/` contains the small installable artifact utility package.
It is not a second simulator. The original top-level package names are retained
because the evaluated code imports `Engine`, `Policies`, `WorldState`, and
`WorldModel` directly.

## Construction of one run

The following chain is common to the direct CLI and paper evaluation:

```text
simulator JSON
  -> Config.config_loader.load_config
  -> policy_registry lookup and policy construction
  -> return planner / pod retriever / path planner injected into assigner
  -> SimulationEngine(...)
  -> WorldState(config)
  -> SimulationEngine.run()
```

`WorldState` in `src/WorldState/world.py` initializes components in a deliberate
order:

1. `MapState` materializes the grid, obstacles, station coordinates, and pod
   homes.
2. `StationState` marks service, queue, buffer, entry, and exit cells before
   robots are placed.
3. `AgentState` objects are created from explicit starts or seeded samples of
   valid free cells.
4. Empty `OrderState` and `TaskState` ledgers are created.
5. The configured pod initializer populates `PodState` and SKU inventory.

Policy names in a JSON file are resolved by
`src/Policies/policy_registry.py`. The registry allows the runner to change an
order generator, task assigner, planner, retriever, or return rule without
forking the engine.

For paper runs, `_run_one_assigner` additionally calls `set_global_seed` for
Python, NumPy, and PyTorch and resets the process-wide task/order ID counters.
When a recorded arrival file is supplied, it switches to
`RecordedOrderGenerator` and disables initial seeding and automatic backlog
refill. The manifest already contains those realized arrivals, so generating
them again would invalidate pairing.

## Exact transition of tick `t`

Before tick 0, `SimulationEngine.run()` invokes `_seed_initial_orders()` if the
configuration requests a nonzero initial pool. Each call to
`SimulationEngine._tick()` then performs the following operations in order:

| Order | Engine operation | Principal implementation |
|---:|---|---|
| 1 | Save tick-start robot positions. | `SimulationEngine._tick` |
| 2 | Generate scheduled arrivals and optionally refill the pending backlog before assignment. | `OrderGenerator.generate`, `_refill_order_backlog` |
| 3 | Materialize fixed order/pod contexts when required and invoke pre-assignment probes. | `Policies/TaskAssigner/context_assignment.py`, `pre_assignment_callbacks` |
| 4 | Call the task assigner and timestamp newly assigned tasks. | `BaseTaskAssigner.assign`, policy-specific assigner |
| 5 | Cascade robots inside station zones, process exits, absorb robots already at an entry, and discard stale entry paths. | `StationState.tick`, engine station helpers |
| 6 | Activate tasks and plan paths. | `_plan_and_activate`; PP calls `PrioritizedPathPlanner.plan` |
| 7 | Move robots by at most one grid edge, absorb new station-entry arrivals, and detect vertex/swap conflicts. | `_move_agents`, `_check_queue_arrivals`, `_detect_conflicts` |
| 8 | Advance pickup, service, and return countdowns; apply completed actions; then test order completion. | `_handle_actions`, `_check_order_completion` |
| 9 | Optionally refill after completion, update stationary-tick state, record metrics, and invoke end-of-tick probes. | `WorldState.risk`, `MetricsTracker`, `on_tick_callbacks` |
| 10 | Optionally render and increment `world.tick`. | renderer, `WorldState.advance_tick` |

Thus an end-of-tick probe for tick `t` observes all actions completed during
that transition while `world.tick` is still `t`. The counter becomes `t+1`
only after metrics and callbacks have run. See [Tick semantics](tick_semantics.md)
for the shorter protocol definition.

## State ownership and lifecycle

`WorldState` is the root object passed to every policy. Its subobjects have
non-overlapping ownership:

| State object | Source file | Owns |
|---|---|---|
| `MapState` | `src/WorldState/map_state.py` | Cell types, walkability, pod homes, and station coordinates. |
| `AgentState` | `src/WorldState/agent_state.py` | Robot position, status, path cursor, carried pod, waits, and stall state. |
| `PodState` / `Pod` | `src/WorldState/pod_state.py` | Pod home/current position, carrier, type, and SKU inventory. |
| `OrderState` / `Order` | `src/WorldState/order_state.py` | Demand, destination station, delivered pods, status, and timestamps. |
| `TaskState` / `Task` | `src/WorldState/task_state.py` | Per-robot `PICK -> DELIVER -> RETURN` chains and task timestamps. |
| `StationState` | `src/WorldState/station_state.py` | Physical slots, admission records, queue-zone motion, and exits. |

The task assigner works through the interface in
`src/Policies/TaskAssigner/base_task_assigner.py`. Its explicit
`propose_assignment_contexts -> select_robots -> commit_assignments` split
separates a fixed `(order, pod, station, return)` context from robot selection.
The action-conditioned world model changes the robot choice for that context;
it does not silently change the order, selected pod, station, or return rule.
The committed unit of work is the lifecycle documented in
[Entities and lifecycle](entities_and_lifecycle.md).

## Orders, pods, and service

The reported stochastic generator is
`src/Policies/OrderGenerator/ZipfOrderGenerator/zipf_order_generator.py`.
It sorts available SKU names, applies the configured Zipf weights, samples
distinct SKUs without replacement, samples integer quantities, and chooses a
station uniformly. `generate()` respects the arrival interval and does not add
a scheduled order at tick 0; `generate_one()` is the unconditional primitive
used for initial-pool and backlog construction.

`RecordedOrderGenerator` in
`src/Policies/OrderGenerator/RecordedOrderGenerator/recorded_order_generator.py`
reconstructs orders at the ticks stored in a JSON manifest. This is the path
used to give every policy in a paired cell exactly the same realized arrivals.

The default pod implementation is split into three independently configured
pieces:

- `DefaultPodInitializer` assigns pod zones to pod types round-robin, samples
  the configured number of SKUs from each type-specific pool, and fills each
  selected SKU with the configured initial quantity.
- `DefaultPodRetriever` greedily adds the available pod that covers the
  largest remaining demand while excluding pods reserved by active tasks and
  pods already delivered for that order.
- `HomeReturnPlanner` returns each pod to its original home cell.

Pickup, station service, and return are not instantaneous. In
`SimulationEngine._handle_actions`, a robot at the action destination starts
the configured integer countdown (`pickup_duration`,
`station_process_duration`, or `dropoff_duration`). The state mutation occurs
only when that countdown reaches zero. A completed delivery marks the pod as
delivered for the order and moves the robot into the station-exit phase; it
does not bypass the queue or return lifecycle.

## Station admission and queue motion

`StationQueueState` in `src/WorldState/station_state.py` distinguishes:

- physical occupancy: service + queue + buffer slots;
- the entry cell, which is a handoff location and is not part of physical
  capacity;
- admitted in-transit robots, tracked separately from physical slots.

The main 50-seed PP protocol selects `physical_only_legacy`. In this mode,
admission is constrained by physical occupancy but does not impose a global
cap on already admitted robots traveling toward a station. At the beginning
of each tick's station phase, `StationState.tick()` cascades queue occupants
toward service and records station observations. Delivery completion and
release are handled by the engine after the configured service countdown.
Alternative committed, FIFO, and ETA admission modes remain in the same source
file for isolated studies, but they are not active in the main comparison.
See [Station admission](station_admission.md) for the experimental contract.

## Dispatch versus path planning

Dispatch chooses which idle robot receives a task chain. Motion is a separate
step after dispatch. For the main PP experiment,
`src/Policies/PathPlanner/PrioritizedPathPlanner/prioritized_path_planner.py`
plans robots sequentially with space-time A*. Paths already accepted in the
tick populate vertex and edge reservations for lower-priority robots. Wait
actions are allowed, and the engine still records realized vertex and swap
conflicts after movement.

The engine also exposes an explicit batch-planning capability for planners
such as PIBT. That branch changes only path activation/planning; it does not
replace order generation, task lifecycle, station processing, or metric
callbacks. Planner choices and parameters are frozen under `configs/planners/`
and in the simulator JSON files.

## Configuration layers

The repository has two complementary configuration layers:

| Layer | Example | Meaning |
|---|---|---|
| Executable simulator JSON | `configs/simulator/world_model_config_PP_48_high.json` | Complete map, robots, pods, action durations, offered-load parameters, and policy constructors. |
| Evaluation contract | `configs/evaluation/main_50seed.yaml` | Campaign-level seeds, 1,500-tick limit, arms, planner/admission contract, manifest pairing, and bootstrap settings. |

The base simulator JSON is reusable and may contain a short default
`max_ticks`; the paper runner overrides it with the campaign value in the
evaluation contract. Running a base JSON directly therefore demonstrates the
simulator but is not, by itself, a reproduction of all 750 main simulations.

For the precise low/mid/high values, see
[Load definitions](../experiments/load_definitions.md). For complete map and
station coordinates, inspect `configs/simulator/`; the concise contracts in
`configs/maps/` summarize the same layouts.

## Metric boundaries

Two metric layers are intentionally kept distinct:

1. `src/Metrics/tracker.py` is the engine-level recorder. It stores completed
   orders, pending queue length, idle robots, realized vertex conflicts,
   robots stationary for at least ten eligible ticks, and measured assignment
   and planning time.
2. The paper runner attaches `OnlineLabelProbe` and calls
   `_extended_sim_metrics` in
   `src/WorldModel/evaluation/evaluate_online_v6.py`. These reuse the training
   label/risk extraction, add open-order ages and task excess delay, and export
   metrics such as `deadlock_ratio_mean` used in the paper tables.

Accordingly, `MetricsTracker.total_deadlock_events` is a count of ticks with at
least one engine-level stalled robot; it must not be substituted for the
paper's mean deadlock ratio. Collapse and station-lock events are downstream
analyses of the exported run/trace data, not hidden simulator states. Their
operational definitions are documented under [Mechanisms](../mechanisms/collapse_definition.md).

## Minimal implementation check

The shortest end-to-end check uses the real configuration loader, policy
registry, world state, planner, and transition loop:

```bash
python scripts/smoke_simulation.py
```

The script materializes the overrides in `configs/evaluation/smoke.json` and
runs `src/main.py` for 20 ticks. It is an implementation smoke test, not a
statistical reproduction. Commands for the full paired campaign are kept in
[Full experiment](../reproduction/full_experiment.md).
