# World model

The model receives a four-frame history of node observations, current directed-edge features, demand context, and one candidate action. A graph encoder produces a latent state; action-conditioned transition and decoder components predict `H=10` future consequences independently for each candidate.

Training supervision is multi-scale: node congestion fields, station queue/load trajectories, seven system labels, candidate ranking, and long-risk continuation targets. `LongRiskHead` is instantiated inside `RMFSWorldModel`; in the canonical trainer it receives a nonzero loss weight during Stage 2 and is optimized in the same run as the action encoder, transition, and decoders. Long-risk labels may be embedded in the training dataset or supplied with `--long-risk-path`, but their separate storage does not create a separate training phase.

The model remains horizon-separated in what it predicts: the transition rolls for `H=10`, while the long-risk head maps the initial and final short-rollout latents to summary statistics of a `W=200` simulator continuation. It does not execute a 200-step latent rollout.

The canonical command is:

```bash
python scripts/train.py world-model --execute \
  --data <world-model-samples.pt> \
  --splits <group-splits.json> \
  --long-risk-path <long-risk-labels.pt> \
  --alpha-long-risk 1.0
```

For auditability, the source tree retains a legacy `train_long_risk_head_only.py` utility used by historical frozen-checkpoint lineages. It is not exposed by the canonical training CLI and should not be used to describe a new from-scratch training run.

Canonical code:

- `src/WorldModel/core/model.py`
- `src/WorldModel/graph/graph_builder.py`
- `src/WorldModel/data/counterfactual_rollout.py`
- `src/WorldModel/training/run_train_v6.py`

Checkpoint bindings are in `checkpoints/checkpoint_manifest.json`.
