# Label-quality audit

## Duplicate-conflict audit (full dataset, 14 feature columns)

- Rows that share all 14 features with at least one other row: 111
- Distinct duplicate feature groups: 54
- Groups with conflicting income labels: **5**
- Rows inside conflicting groups: **10** (listed in `label_conflicts.csv`)

## Data-dictionary re-check of 50 random rows (seed 42)

Rows with any flag: 0 / 50. Rows with an impossible value: 0 / 50.
All 50 rows are in `label_audit_sample.csv` with an `auto_flags` column (rule check) and a `manual_verdict` column (manual check against the data dictionary).

| severity | rule | in sample (50) | in full dataset |
|---|---|---:|---:|
| impossible | age out of 17-90 | 0 | 0 |
| impossible | hours-per-week out of 1-99 | 0 | 0 |
| impossible | capital-gain out of 0-99999 | 0 | 0 |
| impossible | capital-loss out of 0-4356 | 0 | 0 |
| impossible | education-num != code of education | 0 | 0 |
| impossible | Husband but sex=Female | 0 | 1 |
| impossible | Wife but sex=Male | 0 | 3 |
| impossible | Husband/Wife but not married | 0 | 0 |
| impossible | unknown category value | 0 | 0 |
| suspicious | capital-gain and capital-loss both > 0 | 0 | 0 |
| suspicious | Never-worked but hours-per-week > 0 | 0 | 10 |
| suspicious | age < 22 with Masters/Prof-school/Doctorate | 0 | 3 |
| suspicious | hours-per-week = 99 | 0 | 137 |
| suspicious | capital-gain = 99999 (top-code) | 0 | 244 |

## Manual review of the 50-row sample

Reviewed: 50 / 50. OK: 47. Suspicious: 3. Impossible: 0. All labels are plausible for their feature values.

| row_id | verdict |
|---:|---|
| 23881 | suspicious: married but relationship=Own-child (possible if living with parents); label plausible |
| 3375 | suspicious: 4 hours/week is very low for a full-time adult; possible but may be an entry error |
| 47649 | suspicious: married but relationship=Other-relative and >50K is rare for this profile; label not impossible |
