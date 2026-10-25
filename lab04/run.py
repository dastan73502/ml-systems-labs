import argparse
import hashlib
import json
import logging
import platform
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from pipeline.audit import label_audit
from pipeline.card import write_data_card
from pipeline.ingest import ingest, sha256_file
from pipeline.transform import build_preprocessor, drop_train_duplicates, test_overlap
from pipeline.validate import validate

log = logging.getLogger("lab04")


def setup_logging(results):
    Path(results).mkdir(parents=True, exist_ok=True)
    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    for h in (logging.StreamHandler(sys.stdout), logging.FileHandler(Path(results) / "pipeline.log", mode="w")):
        h.setFormatter(fmt)
        log.addHandler(h)


def evaluate(cfg, X_train, y_train, X_test, y_test, derived):
    model = Pipeline([
        ("prep", build_preprocessor(cfg, derived=derived)),
        ("clf", LogisticRegression(max_iter=cfg["baseline"]["max_iter"], C=cfg["baseline"]["C"])),
    ])
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    return {"accuracy": float(accuracy_score(y_test, pred)), "f1": float(f1_score(y_test, pred))}


def save_processed(pre, X, y, path):
    out = pre.transform(X).copy()
    out["income"] = y.values
    out.to_csv(path, index=False, float_format="%.10g", lineterminator="\n")
    return out.shape


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--raw-csv", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    res = Path(cfg["paths"]["results"])
    proc = Path(cfg["paths"]["processed_dir"])
    proc.mkdir(parents=True, exist_ok=True)
    setup_logging(res)
    seed = cfg["seed"]
    np.random.seed(seed)
    target = cfg["source"]["target"]

    log.info("step 1-2: ingestion")
    raw, ingest_info = ingest(cfg, args.raw_csv)

    log.info("step 3: validation")
    df, val = validate(raw, cfg)

    log.info("step 4: stratified split before cleaning")
    X = df.drop(columns=[target])
    y = (df[target] == cfg["source"]["positive_label"]).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["split"]["test_size"], stratify=y if cfg["split"]["stratify"] else None, random_state=seed
    )
    n_train_raw = len(X_train)
    log.info("split: train=%d test=%d positive rate train=%.4f test=%.4f", len(X_train), len(X_test), y_train.mean(), y_test.mean())

    X_train, y_train, n_dupes = drop_train_duplicates(X_train, y_train)
    log.info("dedup: dropped %d exact duplicates from train, train=%d; test untouched=%d", n_dupes, len(X_train), len(X_test))
    overlap = test_overlap(X_train, y_train, X_test, y_test)
    log.info("dedup: %d test rows also appear in train (kept, reported)", overlap)

    pre = build_preprocessor(cfg, derived=True)
    pre.fit(X_train)
    joblib.dump(pre, res / "preprocessor.joblib")
    log.info("transform: fitted on train only, saved %s", res / "preprocessor.joblib")
    clip = pre.named_transformers_["num"].named_steps["clip"]
    bounds = {c: [float(clip.low_[c]), float(clip.high_[c])] for c in clip.columns_}

    train_path, test_path = proc / "train.csv", proc / "test.csv"
    tr_shape = save_processed(pre, X_train, y_train, train_path)
    te_shape = save_processed(pre, X_test, y_test, test_path)
    log.info("transform: processed train %s, test %s", tr_shape, te_shape)

    log.info("step 5: derived features ablation")
    metrics = {
        "without_derived": evaluate(cfg, X_train, y_train, X_test, y_test, derived=False),
        "with_derived": evaluate(cfg, X_train, y_train, X_test, y_test, derived=True),
    }
    for k, v in metrics.items():
        log.info("baseline %-16s accuracy=%.4f f1=%.4f", k, v["accuracy"], v["f1"])
    (res / "metrics.json").write_text(json.dumps(metrics, indent=2))
    d_acc = metrics["with_derived"]["accuracy"] - metrics["without_derived"]["accuracy"]
    d_f1 = metrics["with_derived"]["f1"] - metrics["without_derived"]["f1"]
    (res / "feature_ablation.md").write_text(
        "# Derived features ablation (LogisticRegression, test split)\n\n"
        "| setting | accuracy | F1 (>50K) |\n|---|---:|---:|\n"
        f"| without derived features | {metrics['without_derived']['accuracy']:.4f} | {metrics['without_derived']['f1']:.4f} |\n"
        f"| with capital_net + hours_per_age | {metrics['with_derived']['accuracy']:.4f} | {metrics['with_derived']['f1']:.4f} |\n"
        f"| difference | {d_acc:+.4f} | {d_f1:+.4f} |\n"
    )

    log.info("step 6: label-quality audit")
    audit = label_audit(df, cfg)

    log.info("step 7: hashing, manifest, data card")
    h = hashlib.sha256()
    for p in (train_path, test_path):
        h.update(p.read_bytes())
    hashes = {"train.csv": sha256_file(train_path), "test.csv": sha256_file(test_path), "combined": h.hexdigest()}
    log.info("hash: processed sha256=%s", hashes["combined"])

    manifest = {
        "config": cfg,
        "raw": ingest_info,
        "processed": {
            "train": {"path": str(train_path), "rows": tr_shape[0], "columns": tr_shape[1], "sha256": hashes["train.csv"]},
            "test": {"path": str(test_path), "rows": te_shape[0], "columns": te_shape[1], "sha256": hashes["test.csv"]},
            "combined_sha256": hashes["combined"],
        },
        "split": {"train_before_dedup": n_train_raw, "train_duplicates_dropped": n_dupes, "train": len(X_train), "test": len(X_test), "test_rows_also_in_train": overlap},
        "clip_bounds_from_train": bounds,
        "validation": {"checks": val["checks"], "failed": val["failed"], "rows_dropped": val["rows_dropped"]},
        "label_audit": audit["duplicate_conflict_audit"],
        "metrics": metrics,
        "environment": {"python": platform.python_version(), "pandas": pd.__version__, "numpy": np.__version__, "scikit-learn": sklearn.__version__},
    }
    (res / "processed_manifest.yaml").write_text(yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True))

    write_data_card("data_card.md", {
        "df": df, "target": target, "positive_label": cfg["source"]["positive_label"], "seed": seed,
        "ingest": ingest_info, "n_train_raw": n_train_raw, "n_dupes": n_dupes, "n_train": len(X_train),
        "n_test": len(X_test), "n_features": tr_shape[1] - 1, "metrics": metrics, "hashes": hashes,
    })
    log.info("done: data_card.md, results/processed_manifest.yaml")


if __name__ == "__main__":
    main()
