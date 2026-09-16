# Full experiment

The full path is intentionally separate from lightweight artifact validation. Read the [training pipeline and checkpoint provenance](training_pipeline.md) before running these commands: it distinguishes the supported canonical joint recipe from the head-only repair actually used by historical evaluation checkpoints. Canonical *from-scratch* training can optimize short-horizon decoders and the model-owned long-risk output in one world-model run:

```bash
python scripts/train.py world-model --execute \
  --data <world-model-samples.pt> \
  --splits <group-splits.json> \
  --long-risk-path <long-risk-labels.pt> \
  --alpha-long-risk 1.0
```

J1's station-congestion predictor is trained separately because it is a dispatch auxiliary over a frozen encoder, not part of the world-model loss. Its dataset must first be constructed from matched station traces and frozen latents:

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

6. The fixed-four-station map/fleet experiment has a separate staged runner:

   ```bash
   PYTHONPATH=src python -m WorldModel.evaluation.run_phase_c_density_scale_generalization --help
   ```

   Its `freeze`, `preflight`, `manifest`, `arm`, and `summarize` modes require
   the external model and J1 assets. Use a scratch `--output-root`, not the
   committed artifact directory. The 27 layout/load inputs and 30 complete
   order streams are under `configs/evaluation/density_scale_variants/` and
   `manifests/density_scale_10seed/`; the reviewed, path-free result table
   itself reproduces without those external assets via
   `python scripts/reproduce_density_scale.py`.

7. Export path-free result rows, rerun statistics/figures, and compare with committed artifacts.

The main run is compute intensive (750 arm simulations × 1,500 ticks). Workers affect wall time but not seed/manifests. Do not change checkpoint files, seed sets, detector thresholds, station admission, or load JSON and still call the run a reproduction.
