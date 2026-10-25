import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC = ["age", "fnlwgt", "education-num", "capital-gain", "capital-loss", "hours-per-week"]
CATEGORICAL = ["workclass", "education", "marital-status", "occupation", "relationship", "race", "sex", "native-country"]


class QuantileClipper(BaseEstimator, TransformerMixin):
    def __init__(self, lower=0.01, upper=0.99):
        self.lower = lower
        self.upper = upper

    def fit(self, X, y=None):
        X = pd.DataFrame(X)
        self.columns_ = list(X.columns)
        self.low_ = X.quantile(self.lower)
        self.high_ = X.quantile(self.upper)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_)
        return X.clip(lower=self.low_, upper=self.high_, axis=1)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


class DerivedFeatures(BaseEstimator, TransformerMixin):
    def __init__(self, enabled=True):
        self.enabled = enabled

    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        X = pd.DataFrame(X, columns=self.columns_).copy()
        if self.enabled:
            X["capital_net"] = X["capital-gain"] - X["capital-loss"]
            X["hours_per_age"] = X["hours-per-week"] / (X["age"] - 16).clip(lower=1)
        return X

    def get_feature_names_out(self, input_features=None):
        extra = ["capital_net", "hours_per_age"] if self.enabled else []
        return np.asarray(self.columns_ + extra, dtype=object)


class ToObject(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.columns_ = list(pd.DataFrame(X).columns)
        return self

    def transform(self, X):
        return pd.DataFrame(X, columns=self.columns_).astype(object)

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.columns_, dtype=object)


def build_preprocessor(cfg, derived=True):
    lo, hi = cfg["cleaning"]["clip_quantiles"]
    token = cfg["schema"]["missing_token"]
    numeric = Pipeline([
        ("impute", SimpleImputer(strategy=cfg["cleaning"]["numeric_impute"])),
        ("clip", QuantileClipper(lower=lo, upper=hi)),
        ("derive", DerivedFeatures(enabled=derived)),
        ("scale", StandardScaler()),
    ])
    categorical = Pipeline([
        ("to_object", ToObject()),
        ("impute", SimpleImputer(missing_values=token, strategy=cfg["cleaning"]["categorical_impute"])),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    pre = ColumnTransformer(
        [("num", numeric, NUMERIC), ("cat", categorical, CATEGORICAL)],
        verbose_feature_names_out=False,
    )
    return pre.set_output(transform="pandas")


def drop_train_duplicates(X_train, y_train):
    frame = X_train.assign(_y=y_train.values)
    keep = ~frame.duplicated(keep="first")
    return X_train[keep], y_train[keep], int((~keep).sum())


def test_overlap(X_train, y_train, X_test, y_test):
    tr = X_train.assign(_y=y_train.values)
    te = X_test.assign(_y=y_test.values)
    keys = set(map(tuple, tr.astype(str).values))
    return int(sum(tuple(r) in keys for r in te.astype(str).values))
