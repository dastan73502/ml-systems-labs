import hashlib
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("lab04")

RAW_COLUMNS = [
    "age", "workclass", "fnlwgt", "education", "education-num",
    "marital-status", "occupation", "relationship", "race", "sex",
    "capital-gain", "capital-loss", "hours-per-week", "native-country", "class",
]


def sha256_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_raw(cfg, raw_csv=None):
    if raw_csv:
        log.info("ingest: reading local copy %s", raw_csv)
        return pd.read_csv(raw_csv, header=None, names=RAW_COLUMNS, skipinitialspace=True), f"local:{raw_csv}"
    from sklearn.datasets import fetch_openml
    src = cfg["source"]
    log.info("ingest: fetch_openml(name=%r, version=%s, as_frame=True)", src["name"], src["version"])
    bunch = fetch_openml(name=src["name"], version=src["version"], as_frame=True)
    return bunch.frame, f"openml:{src['name']}/v{src['version']}"


def ingest(cfg, raw_csv=None):
    df, origin = load_raw(cfg, raw_csv)
    rows, cols = df.shape
    log.info("ingest: source=%s rows=%d columns=%d", origin, rows, cols)
    exp = cfg["expected"]
    if (rows, cols) != (exp["rows"], exp["columns"]):
        raise ValueError(f"unexpected shape {rows}x{cols}, expected {exp['rows']}x{exp['columns']}")
    out = Path(cfg["paths"]["raw"])
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    raw_hash = sha256_file(out)
    log.info("ingest: raw snapshot %s sha256=%s", out, raw_hash)
    return df, {"origin": origin, "rows": rows, "columns": cols, "snapshot": Path(out).as_posix(), "sha256": raw_hash}
