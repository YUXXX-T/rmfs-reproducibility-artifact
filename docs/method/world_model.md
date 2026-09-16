# World model

The model receives a four-frame history of node observations, current directed-edge features, demand context, and one candidate action. A graph encoder produces a latent state; action-conditioned transition and decoder components predict `H=10` future consequences independently for each candidate.

Training supervision is multi-scale: node congestion fields, station queue/load trajectories, seven system labels, candidate ranking, and long-risk continuation targets. `LongRiskHead` is instantiated inside `RMFSWorldModel`; the canonical *from-scratch* trainer can optimize it jointly in Stage 2 alongside dynamics and ranking. Long-risk labels may be embedded in the training dataset or supplied with `--long-risk-path`.

The model remains horizon-separated in what it predicts: the transition rolls for `H=10`, while the long-risk head maps the initial and final short-rollout latents to summary statistics of a `W=200` simulator continuation. It does not execute a 200-step latent rollout.

The canonical command is:

```bash
python scripts/train.py world-model --execute \
  --data <world-model-samples.pt> \
  --splits <group-splits.json> \
  --long-risk-path <long-risk-labels.pt> \
  --alpha-long-risk 1.0
```

The [training pipeline](../reproduction/training_pipeline.md) gives the data-generation and optimization details, seed splits, selection rules, and required training assets. This command describes a training recipe; it does not regenerate a frozen evaluated checkpoint without the original inputs.

Canonical code:

- `src/WorldModel/core/model.py`
- `src/WorldModel/graph/graph_builder.py`
- `src/WorldModel/data/counterfactual_rollout.py`
- `src/WorldModel/training/run_train_v6.py`

Checkpoint bindings are in `checkpoints/checkpoint_manifest.json`.
