# Problem definition: income prediction on UCI Adult

**Task.** Binary classification: predict whether a person's annual income exceeds $50K from census attributes. Each record is one adult from the 1994 US Census. The dataset is loaded with `fetch_openml(name='adult', version=2, as_frame=True)` and has 48,842 rows and 15 columns.

**Inputs.** 14 features: 6 integer (age, fnlwgt, education-num, capital-gain, capital-loss, hours-per-week) and 8 categorical (workclass, education, marital-status, occupation, relationship, race, sex, native-country). Missing values are encoded as `?` in workclass, occupation and native-country.

**Output.** `income` in {<=50K, >50K}, mapped to 1 for >50K. The classes are imbalanced: about 24% positive.

**Success metrics.** F1 score for the positive class (>50K) is the primary metric because of the imbalance: a model that always predicts <=50K already reaches about 76% accuracy but has F1 = 0. Accuracy is reported alongside it. Both are measured on a held-out 30% stratified test split that is never used for fitting any preprocessing step. A LogisticRegression baseline sets the reference value, and every pipeline change (for example the derived features) is judged by its change in test F1 and accuracy.

**Constraints.**
- *No leakage.* The train/test split happens before any cleaning. Imputation statistics (median, mode), clipping percentiles, scaling and one-hot categories are fitted on the training split only and then applied to the test split. Duplicate removal is applied to the training split only; the test split stays exactly as sampled.
- *Reproducibility.* One fixed seed (42) in `config.yaml` controls the split, sampling and model. The raw snapshot and the processed data are hashed with SHA256 and stored with the full configuration in `results/processed_manifest.yaml`. The whole pipeline runs with one command (`python run.py`).
- *Transparency.* Validation must report schema, range, missing-value and duplicate problems without silently dropping rows.

**Out of scope.** Model tuning beyond the baseline, and fairness mitigation. Known biases and privacy risks are documented in `data_card.md`.
