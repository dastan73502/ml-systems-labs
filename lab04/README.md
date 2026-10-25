# Lab 4: Data pipeline and quality

Reproducible, leakage-free pipeline for UCI Adult (`fetch_openml(name='adult', version=2, as_frame=True)`).

## Run

```bash
pip install -r requirements.txt
python run.py
```

or `make install && make`. Tests: `make test`.

The first run downloads the dataset from OpenML and caches it in `~/scikit_learn_data`. Offline, a local copy of `adult-all.csv` (15 columns, no header) can be used: `python run.py --raw-csv path/to/adult-all.csv`.

## Steps

| # | Step | Code | Output |
|---|---|---|---|
| 1 | Problem definition | | `problem_definition.md` |
| 2 | Ingestion: read raw data unchanged, log 48842 x 15, snapshot + SHA256 | `pipeline/ingest.py` | `data/raw/adult_raw.csv`, `results/pipeline.log` |
| 3 | Validation: types, ranges, allowed categories, `?` as missing, duplicates, education/education-num consistency; no rows dropped | `pipeline/validate.py`, schema in `config.yaml` | `results/validation_report.md`, `.json` |
| 4 | Stratified 70/30 split before cleaning; train-only dedup; median/mode imputation, 1st/99th percentile clipping, scaling, one-hot fitted on train | `pipeline/transform.py` | `results/preprocessor.joblib`, `data/processed/train.csv`, `test.csv` |
| 5 | Derived features `capital_net`, `hours_per_age`; LogisticRegression with and without | `pipeline/transform.py`, `run.py` | `results/feature_ablation.md`, `results/metrics.json` |
| 6 | Duplicate-conflict audit + data-dictionary re-check of 50 random rows | `pipeline/audit.py` | `results/label_audit.md`, `label_conflicts.csv`, `label_audit_sample.csv` |
| 7 | SHA256 of processed data stored with config; data card | `run.py`, `pipeline/card.py` | `results/processed_manifest.yaml`, `data_card.md` |

## Leakage controls

- The split is the first operation after validation; nothing is fitted before it.
- Imputer, clipping bounds, scaler and one-hot categories are fitted on `X_train` only and stored in one sklearn object (`preprocessor.joblib`).
- Exact duplicates are removed from train only; test rows that also appear in train are counted and kept.
- The ablation fits a fresh preprocessor + model per setting on train and evaluates once on test.

## Reproducibility

- Seed and every parameter live in `config.yaml`.
- `results/processed_manifest.yaml` stores the config, raw and processed SHA256 hashes, split sizes, clipping bounds, metrics and library versions.
- Running `python run.py` twice gives the same processed SHA256 on the same environment.
