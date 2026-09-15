# Tick semantics

A tick is a discrete simulator transition, not wall-clock time. For tick `t`, the engine performs the following ordered operations:

1. generate scheduled arrivals and, in the reported protocol, refill the pending backlog before assignment;
2. materialize fixed order/pod contexts when required, then invoke pre-assignment callbacks;
3. assign task chains;
4. advance station queues, process station exits, absorb prior entry arrivals, and clear stale entry paths;
5. plan/activate tasks and move each robot by at most one cell;
6. absorb new queue arrivals and detect vertex/swap conflicts;
7. apply pickup, delivery, return, and order-completion transitions;
8. update stationary counters, record metrics, and invoke end-of-tick callbacks;
9. render if requested, then increment `world.tick`.

Service and pickup/drop-off waits are integer tick countdowns. A task/order completed during transition `t` stores `completed_at=t`; callbacks at that tick observe the post-action state before `world.tick` advances. The canonical implementation is `src/Engine/simulation_engine.py`.
