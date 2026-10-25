import json
import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger("lab04")


def duplicate_conflicts(df, target):
    feats = [c for c in df.columns if c != target]
    dup = df[df.duplicated(subset=feats, keep=False)]
    groups = dup.groupby(feats, dropna=False)[target].nunique()
    conflict_keys = groups[groups > 1]
    conflict_rows = dup.merge(conflict_keys.reset_index()[feats], on=feats, how="inner")
    return {
        "duplicate_feature_rows": int(len(dup)),
        "duplicate_feature_groups": int(len(groups)),
        "conflicting_groups": int(len(conflict_keys)),
        "conflicting_rows": int(len(conflict_rows)),
    }, conflict_rows


def dictionary_rules(cfg):
    ed_map = cfg["education_num_map"]
    sch = cfg["schema"]["columns"]
    married = {"Married-civ-spouse", "Married-AF-spouse", "Married-spouse-absent"}
    return [
        ("impossible", "age out of 17-90", lambda r: not sch["age"]["min"] <= r["age"] <= sch["age"]["max"]),
        ("impossible", "hours-per-week out of 1-99", lambda r: not sch["hours-per-week"]["min"] <= r["hours-per-week"] <= sch["hours-per-week"]["max"]),
        ("impossible", "capital-gain out of 0-99999", lambda r: not 0 <= r["capital-gain"] <= sch["capital-gain"]["max"]),
        ("impossible", "capital-loss out of 0-4356", lambda r: not 0 <= r["capital-loss"] <= sch["capital-loss"]["max"]),
        ("impossible", "education-num != code of education", lambda r: r["education"] in ed_map and ed_map[r["education"]] != r["education-num"]),
        ("impossible", "Husband but sex=Female", lambda r: r["relationship"] == "Husband" and r["sex"] == "Female"),
        ("impossible", "Wife but sex=Male", lambda r: r["relationship"] == "Wife" and r["sex"] == "Male"),
        ("impossible", "Husband/Wife but not married", lambda r: r["relationship"] in {"Husband", "Wife"} and r["marital-status"] not in married),
        ("impossible", "unknown category value", lambda r: any(r[c] not in set(s["allowed"]) for c, s in sch.items() if s["type"] == "category")),
        ("suspicious", "capital-gain and capital-loss both > 0", lambda r: r["capital-gain"] > 0 and r["capital-loss"] > 0),
        ("suspicious", "Never-worked but hours-per-week > 0", lambda r: r["workclass"] == "Never-worked" and r["hours-per-week"] > 0),
        ("suspicious", "age < 22 with Masters/Prof-school/Doctorate", lambda r: r["age"] < 22 and r["education"] in {"Masters", "Prof-school", "Doctorate"}),
        ("suspicious", "hours-per-week = 99", lambda r: r["hours-per-week"] == 99),
        ("suspicious", "capital-gain = 99999 (top-code)", lambda r: r["capital-gain"] == 99999),
    ]


def check_rows(df, rules):
    flags = []
    for _, r in df.iterrows():
        hits = [f"{sev}: {name}" for sev, name, fn in rules if fn(r)]
        flags.append("; ".join(hits) if hits else "ok")
    return flags


def label_audit(df, cfg):
    out = Path(cfg["paths"]["results"])
    out.mkdir(parents=True, exist_ok=True)
    target = cfg["source"]["target"]
    seed = cfg["seed"]
    n = cfg["label_audit"]["sample_size"]

    conflicts, conflict_rows = duplicate_conflicts(df, target)
    conflict_rows.to_csv(out / "label_conflicts.csv", index=False)

    rules = dictionary_rules(cfg)
    sample = df.sample(n=n, random_state=seed).copy()
    sample.insert(0, "row_id", sample.index)
    sample["auto_flags"] = check_rows(sample, rules)
    sample["manual_verdict"] = ""
    sample.to_csv(out / "label_audit_sample.csv", index=False)

    full_counts = {}
    sample_counts = {}
    for sev, name, fn in rules:
        full_counts[name] = int(df.apply(fn, axis=1).sum())
        sample_counts[name] = int(sample.apply(fn, axis=1).sum())

    flagged = sample[sample["auto_flags"] != "ok"]
    impossible_in_sample = int(sample["auto_flags"].str.contains("impossible").sum())

    report = {
        "duplicate_conflict_audit": conflicts,
        "sample": {"size": n, "seed": seed, "flagged_rows": int(len(flagged)), "impossible_rows": impossible_in_sample},
        "rule_counts_sample": sample_counts,
        "rule_counts_full_dataset": full_counts,
    }
    (out / "label_audit.json").write_text(json.dumps(report, indent=2))

    lines = [
        "# Label-quality audit",
        "",
        "## Duplicate-conflict audit (full dataset, 14 feature columns)",
        "",
        f"- Rows that share all 14 features with at least one other row: {conflicts['duplicate_feature_rows']}",
        f"- Distinct duplicate feature groups: {conflicts['duplicate_feature_groups']}",
        f"- Groups with conflicting income labels: **{conflicts['conflicting_groups']}**",
        f"- Rows inside conflicting groups: **{conflicts['conflicting_rows']}** (listed in `label_conflicts.csv`)",
        "",
        f"## Data-dictionary re-check of {n} random rows (seed {seed})",
        "",
        f"Rows with any flag: {len(flagged)} / {n}. Rows with an impossible value: {impossible_in_sample} / {n}.",
        "All 50 rows are in `label_audit_sample.csv` with an `auto_flags` column and an empty `manual_verdict` column for the manual pass.",
        "",
        "| severity | rule | in sample (50) | in full dataset |",
        "|---|---|---:|---:|",
    ]
    for sev, name, _ in rules:
        lines.append(f"| {sev} | {name} | {sample_counts[name]} | {full_counts[name]} |")
    if len(flagged):
        lines += ["", "### Flagged sample rows", "", "| row_id | flags |", "|---:|---|"]
        for _, r in flagged.iterrows():
            lines.append(f"| {r['row_id']} | {r['auto_flags']} |")
    (out / "label_audit.md").write_text("\n".join(lines) + "\n")
    log.info("audit: conflicting groups=%d rows=%d; sample flagged=%d impossible=%d",
             conflicts["conflicting_groups"], conflicts["conflicting_rows"], len(flagged), impossible_in_sample)
    return report
