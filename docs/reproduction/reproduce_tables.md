# Reproduce tables

```bash
python scripts/reproduce_tables.py
```

This command reads the committed main and six-station per-seed CSV files and
writes:

- main aggregate mean/standard-deviation table;
- complete Proposed–JSQ and Proposed–WM-Base paired intervals;
- six-station aggregate and paired tables;
- collapse count summary;
- runtime summary.

The scale-transfer campaign has a dedicated command because its factorial and
cluster structure differs from the main and six-station analyses:

```bash
python scripts/reproduce_density_scale.py
```

It validates all 1,350 rows and regenerates map/cell tables, paired effects,
four-metric Pareto membership, and two static figures under
`artifacts/generated/`.

Use `--only main`, `--only paired-ci`, or `--only station6` for one group. Bootstrap settings are fixed in `configs/evaluation/main_50seed.yaml` and mirrored in `artifacts/statistics/bootstrap_settings.json`.
