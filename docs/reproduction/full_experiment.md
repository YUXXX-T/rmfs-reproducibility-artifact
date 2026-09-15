# Full experiment

The full path is intentionally separate from lightweight artifact validation. Canonical from-scratch training uses one world-model training run for the short-horizon decoders and the model-owned long-risk output:

```bash
python scripts/train.py world-model --execute \
  --data <world-model-samples.pt> \
  --splits <group-splits.json> \
  --long-risk-path <long-risk-labels.pt> \
  --alpha-long-risk 1.0
```

J1's station-congestion predictor is trained separately because it is a dispatch auxiliary over a frozen encoder, not part of the world-model loss:

```bash
python scripts/train.py j1-dispatch --execute --dataset <station-latents.pt>
```

1. Install the Python 3.10 full environment from `environment.yml` or `requirements.txt`.
2. Download the checkpoint/training assets listed in `checkpoints/checkpoint_manifest.json`.
3. Verify expected files, arrival-manifest schemas, and experiment grids with `python scripts/verify_artifacts.py`.
4. Run the canonical PP module:

   ```bash
   PYTHONPATH=src python -m WorldModel.evaluation.run_phase_c_physical_only_pp_50seed --workers 1
   ```

5. Run the six-station staged adapter:

   ```bash
   PYTHONPATH=src python -m WorldModel.evaluation.run_station6_20x20_adaptation --help
   ```

6. Export path-free result rows, rerun statistics/figures, and compare with committed artifacts.

The main run is compute intensive (750 arm simulations × 1,500 ticks). Workers affect wall time but not seed/manifests. Do not change checkpoint files, seed sets, detector thresholds, station admission, or load JSON and still call the run a reproduction.
