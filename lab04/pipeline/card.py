from pathlib import Path


def pct(part, whole):
    return f"{100 * part / whole:.1f}%"


def write_data_card(path, ctx):
    df = ctx["df"]
    target = ctx["target"]
    n = len(df)
    pos = int((df[target] == ctx["positive_label"]).sum())
    sex = df["sex"].value_counts()
    race = df["race"].value_counts()
    us = int((df["native-country"] == "United-States").sum())
    missing = {c: int((df[c] == "?").sum()) for c in ["workclass", "occupation", "native-country"]}
    rate = df.groupby("sex")[target].apply(lambda s: (s == ctx["positive_label"]).mean())
    qi = ["age", "sex", "race", "native-country"]
    unique_qi = int((df.groupby(qi)[target].transform("size") == 1).sum())
    m = ctx["metrics"]
    h = ctx["hashes"]

    text = f"""# Data card: UCI Adult (Census Income), processed for Lab 4

## Source
- Dataset: Adult / Census Income, extracted by Barry Becker from the 1994 US Census Bureau database; donated to UCI by Ronny Kohavi and Barry Becker (1996).
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML data id 1590), which merges the original `adult.data` and `adult.test` files.
- Ingestion origin for this run: `{ctx['ingest']['origin']}`
- Raw snapshot: `{ctx['ingest']['snapshot']}`, SHA256 `{ctx['ingest']['sha256']}`
- Citation: Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid. KDD-96.

## License
- Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI Machine Learning Repository. Attribution to the donors is required on redistribution.

## Size and composition
- Raw: {ctx['ingest']['rows']} rows x {ctx['ingest']['columns']} columns (14 features + `income`).
- Features: 6 integer (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week), 8 categorical.
- Label: `income` in {{<=50K, >50K}}; positive class >50K = {pos} rows ({pct(pos, n)}).
- Missing (`?`): workclass {missing['workclass']}, occupation {missing['occupation']}, native-country {missing['native-country']}.
- Split: stratified 70/30, seed {ctx['seed']}. Train {ctx['n_train_raw']} rows, {ctx['n_dupes']} exact duplicates removed -> {ctx['n_train']} rows. Test {ctx['n_test']} rows, untouched.
- Processed matrix: {ctx['n_features']} columns after scaling and one-hot encoding (including 2 derived features) + `income` as 0/1.

## Processing
- Validation: schema types, ranges, allowed categories, missing and duplicate checks; violations are reported in `results/validation_report.md`, no rows are dropped.
- Imputation, 1st/99th percentile clipping, scaling and one-hot encoding are fitted on the training split only and saved to `results/preprocessor.joblib`.
- Derived features: `capital_net = capital-gain - capital-loss`, `hours_per_age = hours-per-week / (age - 16)`.
- Baseline LogisticRegression on test: without derived features accuracy {m['without_derived']['accuracy']:.4f}, F1 {m['without_derived']['f1']:.4f}; with derived features accuracy {m['with_derived']['accuracy']:.4f}, F1 {m['with_derived']['f1']:.4f}.
- Processed data SHA256 (train.csv + test.csv): `{h['combined']}`

## Known biases
- Sex imbalance: Male {sex.get('Male', 0)} ({pct(sex.get('Male', 0), n)}), Female {sex.get('Female', 0)} ({pct(sex.get('Female', 0), n)}). Positive rate is {rate.get('Male', 0):.1%} for men and {rate.get('Female', 0):.1%} for women, so a model can learn sex as a strong proxy for income.
- Race imbalance: White {pct(race.get('White', 0), n)}, Black {pct(race.get('Black', 0), n)}, Asian-Pac-Islander {pct(race.get('Asian-Pac-Islander', 0), n)}, Amer-Indian-Eskimo {pct(race.get('Amer-Indian-Eskimo', 0), n)}, Other {pct(race.get('Other', 0), n)}. Minority groups have too few rows for reliable per-group metrics.
- Geography: {pct(us, n)} of rows are United-States; other countries have very small counts.
- Time: 1994 data, the >50K threshold is not adjusted for inflation; the label does not reflect today's income distribution.
- Missingness is not random: `?` in workclass and occupation co-occurs (people outside the labor force), so mode imputation adds bias toward the majority category.
- Top-coding: capital-gain is capped at 99999 and hours-per-week at 99 in the source.
- `fnlwgt` is a census sampling weight, not a personal attribute; using it as a feature is questionable.

## Privacy concerns
- Records are de-identified (no names or addresses), but they describe real people.
- Quasi-identifiers (age, sex, race, native-country) make {unique_qi} rows ({pct(unique_qi, n)}) unique on that combination alone, so linkage with other data could re-identify some individuals.
- Sensitive attributes (race, sex, marital status, native country) are present; any model built on them must be checked for disparate impact before use in decisions about people.
- Intended use: teaching and benchmarking only, not real credit, hiring or benefit decisions.
"""
    Path(path).write_text(text)
