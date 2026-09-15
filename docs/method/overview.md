# Method overview

At a decision opportunity, upstream logic fixes a context `c=(order, pod, station, return_location)`. Candidate actions vary only the idle robot. The world model encodes the state once, rolls each candidate independently for `H=10` latent steps, and predicts node-, station-, and system-level consequences.

The long-risk output head is part of the world model and is optimized with the other world-model objectives during Stage 2. Its targets summarize a declared `W=200` frozen-Greedy simulator continuation; this is label generation, not a second deployed rollout and not a separate canonical model-training stage. S1 uses the resulting candidate consequences to choose a robot within one context.

J1 is dispatch-side. A small auxiliary predictor reads frozen station-node latents and estimates current traffic and service pressure. J1 combines those station channels with exact service debt to rank different contexts once per proposal batch; it never compares context-normalized S1 scores across jobs.

```text
RMFSWorldModel (joint training)
  short-horizon decoders + long-risk output head  ->  S1 robot choice

Frozen WM station representation
  J1 station-congestion predictor                ->  J1 context order
```

The paper’s proposed arm is `ComboS1J1`. `PhaseC` is the same-WM base control without the S1 long-risk conversion and without J1 cross-context ordering.
