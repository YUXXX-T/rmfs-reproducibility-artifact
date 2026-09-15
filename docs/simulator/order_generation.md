# Order generation

The workload generator uses a Zipf distribution with exponent 1.5 over sorted available SKUs. Reported orders contain exactly two distinct SKUs (subject to availability), each with an independently sampled integer quantity from 1 to 3. The destination station is sampled uniformly from station IDs. Pods are initialized with three SKUs each, ten SKUs per pod type, and 50 items per SKU.

Before tick 0 the pool is seeded to 4, 8, or 12 pending orders for low, mid, or high load. During simulation, one scheduled order is attempted every 5, 4, or 3 ticks, respectively. Immediately before assignment, the pending pool is replenished to floors 2, 4, or 6. These mechanisms jointly define offered load; “low/mid/high” is not an empirical quantile label.

For paired evaluation, Greedy records one complete arrival manifest per `(load, seed)` and all other arms replay that exact manifest. The files and their schema/order-count metadata are committed under `manifests/`.
