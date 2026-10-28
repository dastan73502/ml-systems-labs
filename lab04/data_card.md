# Data card: UCI Adult (Census Income), processed for Lab 4

## Source
- Dataset: Adult / Census Income, extracted by Barry Becker from the 1994 US Census Bureau database; donated to UCI by Ronny Kohavi and Barry Becker (1996).
- Loaded with `fetch_openml(name='adult', version=2, as_frame=True)` (OpenML data id 1590), which merges the original `adult.data` and `adult.test` files.
- Ingestion origin for this run: `openml:adult/v2`
- Raw snapshot: `data/raw/adult_raw.csv`, SHA256 `d273d956c28eba72241f69dcbb78e37e93862248503a1873fb44b9fc2a6ae00b`
- Citation: Kohavi, R. (1996). Scaling Up the Accuracy of Naive-Bayes Classifiers: a Decision-Tree Hybrid. KDD-96.

## License
- Creative Commons Attribution 4.0 International (CC BY 4.0), as listed on the UCI Machine Learning Repository. Attribution to the donors is required on redistribution.

## Size and composition
- Raw: 48842 rows x 15 columns (14 features + `income`).
- Features: 6 integer (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week), 8 categorical.
- Label: `income` in {<=50K, >50K}; positive class >50K = 11687 rows (23.9%).
- Missing (`?`): workclass 2799, occupation 2809, native-country 857.
- Split: stratified 70/30, seed 42. Train 34189 rows, 24 exact duplicates removed -> 34165 rows. Test 14653 rows, untouched.
- Processed matrix: 107 columns after scaling and one-hot encoding (including 2 derived features) + `income` as 0/1.

## Processing
- Validation: schema types, ranges, allowed categories, missing and duplicate checks; violations are reported in `results/validation_report.md`, no rows are dropped.
- Imputation, 1st/99th percentile clipping, scaling and one-hot encoding are fitted on the training split only and saved to `results/preprocessor.joblib`.
- Derived features: `capital_net = capital-gain - capital-loss`, `hours_per_age = hours-per-week / (age - 16)`.
- Baseline LogisticRegression on test: without derived features accuracy 0.8540, F1 0.6601; with derived features accuracy 0.8578, F1 0.6696.
- Processed data SHA256 (train.csv + test.csv): `da973bfc97ceb078291109d4f2a0dc8ee9f16d634367743ab51bca7b24905f43`

## Known biases
- Sex imbalance: Male 32650 (66.8%), Female 16192 (33.2%). Positive rate is 30.4% for men and 10.9% for women, so a model can learn sex as a strong proxy for income.
- Race imbalance: White 85.5%, Black 9.6%, Asian-Pac-Islander 3.1%, Amer-Indian-Eskimo 1.0%, Other 0.8%. Minority groups have too few rows for reliable per-group metrics.
- Geography: 89.7% of rows are United-States; other countries have very small counts.
- Time: 1994 data, the >50K threshold is not adjusted for inflation; the label does not reflect today's income distribution.
- Missingness is not random: `?` in workclass and occupation co-occurs (people outside the labor force), so mode imputation adds bias toward the majority category.
- Top-coding: capital-gain is capped at 99999 and hours-per-week at 99 in the source.
- `fnlwgt` is a census sampling weight, not a personal attribute; using it as a feature is questionable.

## Privacy concerns
- Records are de-identified (no names or addresses), but they describe real people.
- Quasi-identifiers (age, sex, race, native-country) make 1500 rows (3.1%) unique on that combination alone, so linkage with other data could re-identify some individuals.
- Sensitive attributes (race, sex, marital status, native country) are present; any model built on them must be checked for disparate impact before use in decisions about people.
- Intended use: teaching and benchmarking only, not real credit, hiring or benefit decisions.
