import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("lab04")


def standardize(raw, cfg):
    df = raw.rename(columns={cfg["source"]["target_raw"]: cfg["source"]["target"]}).copy()
    token = cfg["schema"]["missing_token"]
    for col, spec in cfg["schema"]["columns"].items():
        if col not in df.columns:
            continue
        if spec["type"] == "category":
            s = df[col].astype("object")
            df[col] = s.where(s.notna(), token).astype(str).str.strip()
    return df


def _record(results, check, column, mask, rule):
    idx = list(mask[mask].index[:5])
    n = int(mask.sum())
    results.append({
        "check": check,
        "column": column,
        "rule": rule,
        "violations": n,
        "status": "PASS" if n == 0 else "FAIL",
        "example_rows": [int(i) for i in idx],
        "action": "none" if n == 0 else "reported, rows kept",
    })


def run_checks(df, cfg):
    schema = cfg["schema"]["columns"]
    token = cfg["schema"]["missing_token"]
    results = []

    missing_cols = [c for c in schema if c not in df.columns]
    extra_cols = [c for c in df.columns if c not in schema]
    results.append({
        "check": "columns", "column": "*", "rule": "exact schema columns",
        "violations": len(missing_cols) + len(extra_cols),
        "status": "PASS" if not (missing_cols or extra_cols) else "FAIL",
        "example_rows": [], "action": f"missing={missing_cols} extra={extra_cols}",
    })

    for col, spec in schema.items():
        if col not in df.columns:
            continue
        if spec["type"] == "int":
            num = pd.to_numeric(df[col], errors="coerce")
            _record(results, "type", col, df[col].notna() & num.isna(), "numeric")
            _record(results, "type", col, num.notna() & (num % 1 != 0), "integer-valued")
            _record(results, "missing", col, num.isna(), "no nulls")
            if "min" in spec or "max" in spec:
                lo, hi = spec.get("min", -float("inf")), spec.get("max", float("inf"))
                _record(results, "range", col, num.notna() & ((num < lo) | (num > hi)), f"{lo} <= x <= {hi}")
        else:
            allowed = set(spec["allowed"])
            _record(results, "category", col, ~df[col].isin(allowed), f"in {len(allowed)} allowed values")
            if token in allowed:
                miss = df[col] == token
                results.append({
                    "check": "missing", "column": col, "rule": f"'{token}' allowed, counted",
                    "violations": int(miss.sum()), "status": "INFO",
                    "example_rows": [int(i) for i in miss[miss].index[:5]],
                    "action": "kept, imputed after split",
                })
            else:
                _record(results, "missing", col, df[col] == token, "no missing label")

    ed_map = cfg["education_num_map"]
    expected = df["education"].map(ed_map)
    actual = pd.to_numeric(df["education-num"], errors="coerce")
    _record(results, "consistency", "education/education-num", expected.notna() & (expected != actual), "education-num matches education")

    target = cfg["source"]["target"]
    feats = [c for c in df.columns if c != target]
    dup_rows = df.duplicated(keep="first")
    dup_feats = df.duplicated(subset=feats, keep="first")
    results.append({
        "check": "duplicates", "column": "*", "rule": "exact duplicate rows (all 15 columns)",
        "violations": int(dup_rows.sum()), "status": "INFO",
        "example_rows": [int(i) for i in dup_rows[dup_rows].index[:5]],
        "action": "kept in raw; dropped from train split only",
    })
    results.append({
        "check": "duplicates", "column": "*", "rule": "duplicate feature rows (14 features)",
        "violations": int(dup_feats.sum()), "status": "INFO",
        "example_rows": [int(i) for i in dup_feats[dup_feats].index[:5]],
        "action": "see label audit for conflicting labels",
    })
    return results


def write_report(results, df, out_dir, rows_in):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {
        "rows_in": rows_in,
        "rows_out": len(df),
        "rows_dropped": rows_in - len(df),
        "checks": len(results),
        "failed": sum(r["status"] == "FAIL" for r in results),
        "results": results,
    }
    (out_dir / "validation_report.json").write_text(json.dumps(summary, indent=2))
    lines = [
        "# Validation report",
        "",
        f"Rows in: {rows_in}  |  Rows out: {len(df)}  |  Rows dropped: {rows_in - len(df)}",
        f"Checks run: {summary['checks']}  |  Failed: {summary['failed']}",
        "",
        "| check | column | rule | violations | status | action | example rows |",
        "|---|---|---|---:|---|---|---|",
    ]
    for r in results:
        lines.append(
            f"| {r['check']} | {r['column']} | {r['rule']} | {r['violations']} | {r['status']} | {r['action']} | {r['example_rows']} |"
        )
    (out_dir / "validation_report.md").write_text("\n".join(lines) + "\n")
    return summary


def validate(raw, cfg):
    df = standardize(raw, cfg)
    for col, spec in cfg["schema"]["columns"].items():
        if spec["type"] == "int":
            df[col] = pd.to_numeric(df[col], errors="coerce")
    results = run_checks(df, cfg)
    summary = write_report(results, df, cfg["paths"]["results"], len(raw))
    if len(df) != len(raw):
        raise RuntimeError("validation must not drop rows")
    for r in results:
        if r["status"] != "PASS":
            log.info("validate: %-11s %-24s %-38s %6d %s", r["check"], r["column"], r["rule"], r["violations"], r["status"])
    log.info("validate: %d checks, %d failed, rows kept %d/%d", summary["checks"], summary["failed"], len(df), len(raw))
    return df, summary
