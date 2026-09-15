# Reproduce tables

```bash
python scripts/reproduce_tables.py
```

This command reads the two committed per-seed CSV files and writes:

- main aggregate mean/standard-deviation table;
- complete Proposed–JSQ and Proposed–WM-Base paired intervals;
- six-station aggregate and paired tables;
- collapse count summary;
- runtime summary.

Use `--only main`, `--only paired-ci`, or `--only station6` for one group. Bootstrap settings are fixed in `configs/evaluation/main_50seed.yaml` and mirrored in `artifacts/statistics/bootstrap_settings.json`.
