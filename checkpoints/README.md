# Checkpoints

Large model binaries are not committed to Git. `checkpoint_manifest.json` records each release asset's logical name, expected path, distribution status, and byte size when known. Some expected paths retain historical experiment-directory names so a reviewer can place the frozen evaluated files without renaming them; those path labels are provenance, not the canonical from-scratch training interface.

During anonymous review, obtain the release bundle supplied with the anonymous repository and place each file at its listed path. After public release, populate the currently null URLs with GitHub Release or Zenodo links. Then run:

```bash
python scripts/download_artifacts.py --list
python scripts/verify_checkpoints.py
```

The statistics-only reproduction does not require these files. For full evaluation, use the named files from the review release bundle; do not substitute differently trained local models.
