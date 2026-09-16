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

The paper's frozen evaluation checkpoint lineage is a separate historical fact: the six-station runner explicitly trains its core with long-risk loss set to zero, then runs `train_long_risk_head_only.py` on seed-disjoint continuation labels. The four-station release manifest also identifies the evaluated long-risk checkpoint as a head-only repair. Thus the canonical joint recipe must not be presented as proof that every evaluated checkpoint was jointly trained. The head remains *part of the world model at inference* in either case. See the [training pipeline](../reproduction/training_pipeline.md) for data, stages, seeds, selection, and availability.

Canonical code:

- `src/WorldModel/core/model.py`
- `src/WorldModel/graph/graph_builder.py`
- `src/WorldModel/data/counterfactual_rollout.py`
- `src/WorldModel/training/run_train_v6.py`

Checkpoint bindings are in `checkpoints/checkpoint_manifest.json`.
