# Frozen-Greedy continuation

Counterfactual labels must specify what happens after a candidate action is committed. The continuation protocol freezes the evaluated one-step intervention, then lets the unmodified Greedy task assigner control subsequent dispatch opportunities while the simulator continues with the same planner, station rules, and future arrival stream.

For the model-owned long-risk head, this simulator continuation lasts 200 ticks; targets include peak/tail/terminal risk statistics and are consumed by the joint world-model trainer. Complete behavior-continuation trajectories include future order generation and the continuation scheduler. Right-censored trajectories are rejected.

“Frozen Greedy” means the continuation policy and its parameters are fixed, not that the physical state is frozen. Robots, stations, queues, service, returns, and arrivals evolve normally. The implementation is `src/WorldModel/data/counterfactual_rollout.py`; the audited seed contracts are in `configs/training` and `splits/`.
