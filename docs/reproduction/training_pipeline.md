# Training pipeline and seed splits

This page describes the world-model training recipe, the separate J1
dispatch-side predictor, the data and seed partitions, and what is required
to rerun training. A new training run does not reproduce the paper's exact
evaluated weights without its original inputs and checkpoint selection.

## Data and supervision

A training example starts from four observation frames, current graph edges
and demand context, and one candidate robot assignment in an otherwise fixed
`(order, pod, station, return)` context. The simulator commits the candidate,
then rolls forward to generate `H=10` node, station and seven system-label
trajectories. Candidate actions sharing a decision context form a ranking
group. Use `candidate_group_id`/source-seed partitions for train, validation,
and test; random splitting of individual candidates would leak the same
decision context across partitions. The [feature and label schema](../method/state_action_labels.md)
gives the exact channel order.

Long-risk supervision has a different time scale. After that one-step
intervention, the simulator continues for `W=200` ticks under frozen Greedy
dispatch, including future arrivals, service, and returns. Complete, uncensored
continuations produce the risk-tail and terminal labels for the model-owned
`LongRiskHead`; these are **target-generation simulations**, not a 200-step
latent rollout at inference. The head predicts six outputs described in
`schemas/long_risk_labels.yaml`, while the short dynamics model still rolls
only ten steps. `src/WorldModel/data/counterfactual_rollout.py` implements the
continuation and `src/WorldModel/data/generate_long_risk_labels.py` prepares
long-risk targets.

Auditable scale from the source training summaries, exported without machine
paths or fingerprints in `artifacts/statistics/training_data_scale.json` (these
counts describe the **training split** of the named stages, not an aggregate
over later fine-tuning stages):

| Source stage | Train candidates | Train groups | Available train pairs |
|---|---:|---:|---:|
| Four-station `model_round1_v1` core | 7,915 | 902 | 23,979 |
| Six-station `full_711_740_v1/training/world_model_v1` base | 10,810 | 1,420 | 32,110 |

Available pairs describe the split before the trainer's configurable
`--max-train-pairs` cap; they are not a count of gradient steps.

The four-station J1 *development* latent dataset records 86,112 train,
28,704 validation and 28,704 test station rows across 120 source runs. It is
explicitly marked `development_only` in its source summary; those row counts
must **not** be misreported as the formal training scale of the released
J1 checkpoint. Exact counts for other training stages are not available in
the compact files committed here; the source tensors and complete training
logs are distributed separately.

## Canonical joint world-model recipe

`scripts/train.py world-model` delegates to
`src/WorldModel/training/run_train_v6.py`. Its defaults are a two-stage
*recipe*, not a statement about the frozen paper-checkpoint hyperparameters:

1. **Stage 1 — dynamics warm-up.** Truncate targets to `H=3`; train the short
   node, station, and system predictions without pairwise ranking or
   long-risk loss (`alpha_long_risk=0`). Default: 15 epochs, Adam, learning
   rate `1e-3`.
2. **Stage 2 — joint training.** Extend to `H=10`; optimize dynamics, candidate
   ranking and available long-risk labels in the same world-model checkpoint.
   Default: up to 30 epochs, learning rate `3e-4`, ranking weight `0.1`,
   long-risk weight `1.0`. The first five epochs freeze the state/demand
   encoders and short-horizon decoders while updating the action encoder,
   transition, and long-risk head. The default subsequent unfreeze mode trains
   all modules at 0.3 times the Stage-2 learning rate. Validation ranking
   accuracy governs early stopping with patience eight by default; an
   alternative top-1-regret monitor is available.

If `--alpha-long-risk` is positive, the trainer requires long-risk labels in
the training split and fails early when they are absent. Labels can already
be embedded in `--data`, or supplied in a separate tensor file with
`--long-risk-path` and merged at load time. A separate *label file* is not a
separate *deployed model*. Supply `--splits` from the fused dataset to enforce
the declared group/seed partition; omitting it invokes the trainer's
group-based fallback split, which is not interchangeable with a frozen paper
split. Stage 1 saves `stage1_world_model.pt`; Stage 2 saves
`world_model.pt`, best-validation-ranking and best-regret checkpoints, epoch
logs, and `train_summary.json`. The exact checkpoint chosen for evaluation
must be recorded explicitly.

```bash
python scripts/train.py world-model --execute \
  --data <fused-world-model-samples.pt> \
  --splits <fused-splits.json> \
  --long-risk-path <continuation-labels.pt> \
  --alpha-long-risk 1.0 \
  --save-dir artifacts/generated/checkpoints/world_model_v6
```

The published YAML contract `configs/training/world_model.yaml` describes
this joint recipe and its model-owned long-risk output.

## Six-station adaptation and held-out evaluation

The six-station study adapts the demand representation from `5+4=9` to
`5+6=11`, builds six-station training data, and evaluates held-out policy
runs. The world-model core data and long-risk/J1 auxiliary-label data use
different seed blocks. This study is **adaptation**, not four-to-six-station
zero-shot transfer; the training and evaluation split is recorded in
`configs/training/station6_adaptation.yaml`.

| Six-station use | Train seeds | Validation | Offline test | Online evaluation |
|---|---|---|---|---|
| Behavior/Phase-C core | 711–717 | 718–719 | 720 | — |
| Long-risk labels and J1 auxiliary | 731–737 | 738–739 | 740 | — |
| Final paired policy test | — | — | — | 721–730 |

These blocks do not overlap. The executable stage definitions are in
`src/WorldModel/evaluation/run_station6_20x20_adaptation.py`. The separate
701–710 map/fleet-scale experiment **does not train**: it holds the WM,
long-risk output and J1 predictor fixed.

## J1 predictor: a separate dispatch-side training job

J1's two-channel station-congestion predictor estimates **current-state**
`traffic` and `service` pressure from frozen station-node representations.
It is neither the WM's future-station-trajectory decoder nor its long-risk
head. Build the dataset by aligning saved station traces to the same-tick WM
inputs, running a frozen encoder, and fitting target scales on training seeds
only. The four-station development split uses 511–516 for training, 517–518
for validation, and 519–520 for offline testing across low/mid/high loads
and Greedy, Hungarian, PhaseC and PhaseC-S1 source arms.

`src/WorldModel/data/build_station_congestion_head_dataset.py` creates
station latents, targets, split metadata, and scale contract. The separate
trainer `src/WorldModel/training/train_station_congestion_head.py` fits only
the small station predictor with equal-channel MSE and AdamW; it selects the
best validation-MSE epoch (default learning rate `1e-3`, batch size 4096,
patience 15). The encoder is not updated. At dispatch time the predicted
channels are combined with **exact**, non-learned service debt, as detailed
in [J1 context dispatch](../method/j1_context_dispatch.md).

```bash
PYTHONPATH=src python -m WorldModel.data.build_station_congestion_head_dataset \
  --input-root <saved-station-traces> \
  --checkpoint <frozen-world-model.pt> \
  --output-root artifacts/generated/j1_dataset
python scripts/train.py j1-dispatch --execute \
  --dataset artifacts/generated/j1_dataset/station_congestion_latents.pt \
  --output-root artifacts/generated/checkpoints/j1_dispatch
```

These commands require unbundled trace/tensor and checkpoint inputs. Do not
pass `--formal` to a dataset marked `development_only`; a formal J1 dataset
must be prepared and audited separately. For six-station adaptation, the
auxiliary predictor instead uses the disjoint 731–740 block and the adapted
six-station encoder. The released J1 checkpoint and its normalization
contract are separate from the WM checkpoint and must be kept paired.

## What can be reproduced from this checkout

The full training code, seed contracts, label schemas and lightweight
evaluation tables are here. Large world-model samples, `W=200` continuation
labels, station-latent training tensors, selected checkpoint binaries, and
complete training logs are **not** committed. The repository's
`checkpoints/README.md` describes their planned separate distribution; the
[artifact overview](../artifact_overview.md) also lists what is missing. Without those assets, this
checkout supports inspection of the training procedure and reproduction of
published statistics, **not** independent from-scratch retraining or exact
weight matching. The [full experiment guide](full_experiment.md) describes
the subsequent evaluation stages.
