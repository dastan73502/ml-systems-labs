from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from pipeline.transform import build_preprocessor, drop_train_duplicates
from pipeline.validate import validate

ROOT = Path(__file__).resolve().parents[1]


def cfg(tmp_path):
    c = yaml.safe_load((ROOT / "config.yaml").read_text())
    c["paths"]["results"] = str(tmp_path)
    return c


def sample_frame():
    row = {
        "age": 39, "workclass": "State-gov", "fnlwgt": 77516, "education": "Bachelors", "education-num": 13,
        "marital-status": "Never-married", "occupation": "Adm-clerical", "relationship": "Not-in-family",
        "race": "White", "sex": "Male", "capital-gain": 2174, "capital-loss": 0, "hours-per-week": 40,
        "native-country": "United-States", "class": "<=50K",
    }
    return pd.DataFrame([row] * 6)


def by_rule(summary):
    return {(r["check"], r["column"]): r["violations"] for r in summary["results"] if r["status"] == "FAIL"}


def test_validation_reports_violations_and_keeps_rows(tmp_path):
    raw = sample_frame()
    raw.loc[0, "age"] = 12
    raw.loc[1, "hours-per-week"] = 120
    raw.loc[2, "capital-loss"] = 5000
    raw.loc[3, "workclass"] = "Freelance"
    raw.loc[4, "education-num"] = 9
    raw.loc[5, "class"] = "?"
    df, summary = validate(raw, cfg(tmp_path))
    fails = by_rule(summary)
    assert len(df) == len(raw)
    assert summary["rows_dropped"] == 0
    assert fails[("range", "age")] == 1
    assert fails[("range", "hours-per-week")] == 1
    assert fails[("range", "capital-loss")] == 1
    assert fails[("category", "workclass")] == 1
    assert fails[("consistency", "education/education-num")] == 1
    assert fails[("missing", "income")] == 1
    assert (tmp_path / "validation_report.md").exists()


def test_question_mark_and_nan_are_missing(tmp_path):
    raw = sample_frame()
    raw.loc[0, "workclass"] = "?"
    raw.loc[1, "workclass"] = np.nan
    df, summary = validate(raw, cfg(tmp_path))
    assert (df["workclass"] == "?").sum() == 2
    assert summary["failed"] == 0


def test_preprocessor_uses_train_statistics_only(tmp_path):
    c = cfg(tmp_path)
    train = sample_frame().drop(columns=["class"])
    train["age"] = [20, 30, 40, 50, 60, 70]
    test = train.copy()
    test["age"] = [90, 90, 90, 90, 90, 90]
    pre = build_preprocessor(c).fit(train)
    clip = pre.named_transformers_["num"].named_steps["clip"]
    before = clip.high_["age"]
    pre.transform(test)
    assert clip.high_["age"] == before
    assert clip.high_["age"] < 90


def test_dedup_only_touches_given_split():
    X = sample_frame().drop(columns=["class"])
    y = pd.Series([0] * 6)
    Xd, yd, n = drop_train_duplicates(X, y)
    assert n == 5 and len(Xd) == 1 and len(yd) == 1
