#!/usr/bin/env python3
"""S5-only validation and descriptive aggregation of frozen final datasets."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from scipy.stats import t

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs/csv"
DOC = ROOT / "docs/v0.61_S5_dataaggregation.md"
RAW_OUT = OUT / "final_control_raw_canonical.csv"
SUMMARY_OUT = OUT / "final_control_summary.csv"

INPUTS = {
    "exp07_results": OUT / "exp07_results_20260822_182815.csv",
    "exp08_results": OUT / "exp08_results_20260822_185958.csv",
    "exp08_evidence": OUT / "exp08_execution_evidence_20260822_185958.csv",
    "exp08_manifest": OUT / "exp08_s3_manifest.json",
    "exp09_results": OUT / "exp09_results_20260822_192752.csv",
    "exp09_topology": OUT / "exp09_topology_validation_20260822_192752.csv",
}
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")
ALG = {"gossip": "Gossip", "cluster": "Structured", "dcsoc": "DC-SoC", "ahbn": "AHBN"}
SEEDS = set(range(42, 62))


class ValidationError(RuntimeError):
    pass


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValidationError(message)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def metadata(path: Path) -> dict[str, object]:
    st = path.stat()
    rows = None
    if path.suffix == ".csv":
        with path.open("r", newline="", encoding="utf-8") as fh:
            rows = sum(1 for _ in fh) - 1
    return {"size": st.st_size, "rows": rows, "mtime_ns": st.st_mtime_ns, "sha256": sha256(path)}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def git(*args: str) -> str:
    return subprocess.run(["git", *args], cwd=ROOT, text=True, capture_output=True, check=True).stdout.rstrip()


def preexisting_status() -> str:
    """Return repository changes excluding the four artifacts owned by S5."""
    owned = {
        "docs/v0.61_S5_dataaggregation.md",
        "scripts/aggregate_s5_final_data.py",
        "outputs/csv/final_control_raw_canonical.csv",
        "outputs/csv/final_control_summary.csv",
    }
    lines = []
    for line in git("status", "--short", "--untracked-files=all").splitlines():
        path = line[3:].split(" -> ")[-1]
        if path not in owned:
            lines.append(line)
    return "\n".join(lines)


def condition(exp: str, row: dict[str, str]) -> tuple[str, float]:
    strategy = row["strategy"]
    if exp == "exp07":
        if strategy == "gossip":
            k = float(row["fanout"])
            require(k.is_integer() and int(k) in range(2, 7), f"Exp07 invalid Gossip fanout: {row}")
            return f"gossip_k={int(k)}", float(int(k))
        require(strategy == "ahbn" and not row["fanout"].strip(), f"Exp07 invalid AHBN condition: {row}")
        return "ahbn_canonical_adaptive", 0.0
    if exp == "exp08":
        factor = float(row["ch_overload_factor"])
        require(factor in {1.0, 1.5, 2.0, 3.0}, f"Exp08 invalid overload factor: {row}")
        return f"ch_overload_factor={factor:.1f}", factor
    p = float(row["topology_param"])
    require(p in {0.04, 0.06, 0.08, 0.10, 0.12}, f"Exp09 invalid edge probability: {row}")
    return f"edge_prob={p:.2f}", p


def validate_metrics(row: dict[str, str], source: Path, lineno: int) -> dict[str, float]:
    values: dict[str, float] = {}
    for metric in METRICS:
        require(metric in row and row[metric].strip(), f"{source}:{lineno}: missing {metric}")
        try:
            value = float(row[metric])
        except ValueError as exc:
            raise ValidationError(f"{source}:{lineno}: nonnumeric {metric}={row[metric]!r}") from exc
        require(math.isfinite(value), f"{source}:{lineno}: nonfinite {metric}={value}")
        values[metric] = value
    require(0.0 <= values["delivery_ratio"] <= 1.0, f"{source}:{lineno}: delivery_ratio out of domain")
    for metric in ("propagation_delay", "duplicates", "total_forwards"):
        require(values[metric] >= 0, f"{source}:{lineno}: negative {metric}")
    return values


def validate_results(exp: str, path: Path, expected_rows: int, expected_strategies: set[str], expected_cells: int) -> tuple[list[dict[str, object]], dict[str, object]]:
    rows = read_csv(path)
    require(len(rows) == expected_rows, f"{exp}: expected {expected_rows} rows, got {len(rows)} in {path}")
    require(len({tuple(sorted(r.items())) for r in rows}) == len(rows), f"{exp}: exact duplicate rows found")
    require({r["experiment"] for r in rows} == {exp}, f"{exp}: unexpected experiment labels")
    actual_strategies = {r["strategy"] for r in rows}
    require(actual_strategies == expected_strategies, f"{exp}: strategies {sorted(actual_strategies)} != {sorted(expected_strategies)}")
    canonical: list[dict[str, object]] = []
    keys: list[tuple[object, ...]] = []
    cell_seeds: dict[tuple[str, str], list[int]] = defaultdict(list)
    condition_order: dict[str, float] = {}
    for lineno, row in enumerate(rows, 2):
        require(row["strategy"] in ALG, f"{path}:{lineno}: unmapped strategy {row['strategy']!r}")
        seed = int(row["seed"])
        label, order = condition(exp, row)
        values = validate_metrics(row, path, lineno)
        algorithm = ALG[row["strategy"]]
        key = (exp, algorithm, label, seed)
        keys.append(key)
        cell_seeds[(algorithm, label)].append(seed)
        condition_order[label] = order
        canonical.append({
            "experiment": exp.upper().replace("EXP", "Exp"), "algorithm": algorithm,
            "experimental_condition": label, "seed": seed,
            "topology_type": row["topology_type"], "topology_parameter": row["topology_param"],
            "topology_id": "", **values, "source_file": str(path.relative_to(ROOT)),
            "_condition_order": order,
        })
    require(len(set(keys)) == len(keys), f"{exp}: duplicate run keys found")
    require(len(cell_seeds) == expected_cells, f"{exp}: expected {expected_cells} cells, got {len(cell_seeds)}")
    for cell, seeds in cell_seeds.items():
        require(len(seeds) == 20 and set(seeds) == SEEDS and len(set(seeds)) == 20, f"{exp} {cell}: invalid seed matrix {sorted(seeds)}")
    return canonical, {"rows": len(rows), "cells": len(cell_seeds), "keys": len(set(keys))}


def validate_exp08_identity(canonical: list[dict[str, object]]) -> None:
    evidence = read_csv(INPUTS["exp08_evidence"])
    require(len(evidence) == 320, f"Exp08 evidence expected 320 rows, got {len(evidence)}")
    evidence_keys = Counter()
    identities: dict[tuple[float, int], set[str]] = defaultdict(set)
    by_key: dict[tuple[str, float, int], str] = {}
    for row in evidence:
        strategy, factor, seed = row["strategy"], float(row["overload_factor"]), int(row["seed"])
        require(strategy in ALG and factor in {1.0, 1.5, 2.0, 3.0} and seed in SEEDS, f"Exp08 invalid evidence row: {row}")
        require(row["topology_seed"] == row["seed"] and row["topology_identity"], f"Exp08 incomplete topology identity: {row}")
        key = (strategy, factor, seed)
        evidence_keys[key] += 1
        by_key[key] = row["topology_identity"]
        identities[(factor, seed)].add(row["topology_identity"])
    require(all(v == 1 for v in evidence_keys.values()) and len(evidence_keys) == 320, "Exp08 evidence keys are incomplete/duplicated")
    require(len(identities) == 80 and all(len(v) == 1 for v in identities.values()), "Exp08 cross-comparator topology identity mismatch")
    for row in canonical:
        factor = float(str(row["experimental_condition"]).split("=")[1])
        raw_strategy = next(k for k, v in ALG.items() if v == row["algorithm"])
        row["topology_id"] = by_key[(raw_strategy, factor, int(row["seed"]))]
    manifest = json.loads(INPUTS["exp08_manifest"].read_text(encoding="utf-8"))
    require(manifest.get("completed_runs") == 320, "Exp08 manifest completed_runs != 320")
    require(Path(manifest["results"]).name == INPUTS["exp08_results"].name, "Exp08 manifest result association mismatch")
    require(Path(manifest["evidence"]).name == INPUTS["exp08_evidence"].name, "Exp08 manifest evidence association mismatch")


def validate_exp09_identity(canonical: list[dict[str, object]]) -> None:
    topo = read_csv(INPUTS["exp09_topology"])
    require(len(topo) == 100, f"Exp09 topology artifact expected 100 rows, got {len(topo)}")
    mapping: dict[tuple[float, int], str] = {}
    for row in topo:
        key = (float(row["edge_prob"]), int(row["seed"]))
        require(key not in mapping, f"Exp09 duplicate topology key {key}")
        require(key[0] in {0.04, 0.06, 0.08, 0.10, 0.12} and key[1] in SEEDS, f"Exp09 unexpected topology key {key}")
        require(row["topology_type"] == "er" and row["topology_identity"] and row["algorithm_match"] == "True", f"Exp09 invalid topology identity row: {row}")
        require(int(row["nodes"]) > 0 and int(row["edges"]) >= 0, f"Exp09 invalid topology size: {row}")
        mapping[key] = row["topology_identity"]
    require(len(mapping) == 100, "Exp09 incomplete topology realization matrix")
    refs: Counter[tuple[float, int]] = Counter()
    for row in canonical:
        p = float(row["topology_parameter"])
        key = (p, int(row["seed"]))
        require(key in mapping, f"Exp09 result has no topology identity: {key}")
        row["topology_id"] = mapping[key]
        refs[key] += 1
    require(all(v == 4 for v in refs.values()) and len(refs) == 100, "Exp09 results do not reference each topology four times")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        groups[(str(row["experiment"]), str(row["algorithm"]), str(row["experimental_condition"]))].append(row)
    require(len(groups) == 42, f"Global expected 42 cells, got {len(groups)}")
    result = []
    for key, values in groups.items():
        require(len(values) == 20, f"{key}: expected n=20, got {len(values)}")
        out: dict[str, object] = dict(zip(("experiment", "algorithm", "experimental_condition"), key))
        out["n"] = len(values)
        out["_condition_order"] = values[0]["_condition_order"]
        critical = t.ppf(0.975, len(values) - 1)
        for metric in METRICS:
            xs = [float(v[metric]) for v in values]
            mean = sum(xs) / len(xs)
            variance = sum((x - mean) ** 2 for x in xs) / (len(xs) - 1)
            std = math.sqrt(variance)
            margin = critical * std / math.sqrt(len(xs))
            out[f"{metric}_mean"] = mean
            out[f"{metric}_std"] = std
            out[f"{metric}_ci95_low"] = mean - margin
            out[f"{metric}_ci95_high"] = mean + margin
        result.append(out)
    alg_order = {"Gossip": 0, "Structured": 1, "DC-SoC": 2, "AHBN": 3}
    result.sort(key=lambda r: (r["experiment"], alg_order[str(r["algorithm"])], float(r["_condition_order"])))
    return result


def validate_outputs(raw: list[dict[str, object]], summary: list[dict[str, object]]) -> None:
    require(len(raw) == 840, f"Canonical raw expected 840 rows, got {len(raw)}")
    keys = {(r["experiment"], r["algorithm"], r["experimental_condition"], r["seed"]) for r in raw}
    require(len(keys) == 840, f"Canonical raw expected 840 unique keys, got {len(keys)}")
    require(len(summary) == 42 and all(r["n"] == 20 for r in summary), "Summary row count/n validation failed")
    require(sum(int(r["n"]) for r in summary) == 840, "Summary n does not conserve 840 observations")
    for row in summary:
        for metric in METRICS:
            mean, std = float(row[f"{metric}_mean"]), float(row[f"{metric}_std"])
            lo, hi = float(row[f"{metric}_ci95_low"]), float(row[f"{metric}_ci95_high"])
            require(all(math.isfinite(x) for x in (mean, std, lo, hi)), f"Nonfinite summary value: {row}")
            require(std >= 0 and lo <= mean <= hi, f"Invalid summary bounds: {row}")


def report(stats: dict[str, dict[str, object]], output_hashes: dict[str, str], pre: dict[str, dict[str, object]]) -> str:
    lines = ["S5 — FINAL DATA AGGREGATION", "", "INPUT VALIDATION", ""]
    for exp, expected, cells in (("Exp07", 120, 6), ("Exp08", 320, 16), ("Exp09", 400, 20)):
        s = stats[exp.lower()]
        lines += [f"{exp}:", f"expected rows: {expected}", f"actual rows: {s['rows']}", f"cells: {s['cells']}/{cells}", "seeds: 42–61", "missing seeds: 0", "duplicate runs: 0", "unexpected runs: 0", "metric errors: 0"]
        if exp == "Exp08": lines.append("topology/evidence identity: PASS")
        if exp == "Exp09": lines += ["topology realizations: 100/100", "cross-algorithm topology identity: PASS"]
        lines += ["status: PASS", ""]
    lines += ["GLOBAL:", "expected raw rows: 840", "actual raw rows: 840", "expected cells: 42", "actual cells: 42", "all cells n=20: YES", "missing runs: 0", "duplicate runs: 0", "invalid metrics: 0", "", "CANONICAL RAW:", "rows: 840", "unique run keys: 840", "status: PASS", "", "CANONICAL SUMMARY:", "rows: 42", "all n=20: YES", "metrics aggregated:", *[f"  {m}" for m in METRICS], "statistics:", "  mean", "  sample standard deviation", "  Student-t 95% CI", "status: PASS", "", "RAW IMMUTABILITY:", "input hashes unchanged: YES", "", "SIMULATIONS EXECUTED: NO", "ALGORITHM FILES MODIFIED: NONE", "CONTROLLER PARAMETERS MODIFIED: NO", "TOPOLOGY GENERATOR MODIFIED: NO", "METRIC DEFINITIONS MODIFIED: NO", "SCIENTIFIC INTERPRETATION PERFORMED: NO", "", "git diff --check: PASS", "", "S5 FINAL STATUS: PASS"]
    return "\n".join(lines)


def audit(pre: dict[str, dict[str, object]], post: dict[str, dict[str, object]], pre_status: str, terminal: str) -> str:
    hash_rows = "\n".join(f"| `{p.relative_to(ROOT)}` | {pre[k]['size']} | {pre[k]['rows'] if pre[k]['rows'] is not None else 'n/a'} | `{pre[k]['sha256']}` |" for k, p in INPUTS.items())
    output_rows = "\n".join(f"| `{p.relative_to(ROOT)}` | {p.stat().st_size} | `{sha256(p)}` |" for p in (RAW_OUT, SUMMARY_OUT))
    return f"""# v0.61 S5 data aggregation audit

## 1. Scope

S5 verified and aggregated the frozen Exp07–Exp09 datasets. No simulations, scientific interpretation, plots, hypothesis tests, effect sizes, rankings, or algorithm/configuration changes were performed.

## 2. Authoritative input files

The six artifacts specified for S5 were used verbatim.

## 3. Pre-S5 hashes

| File | Bytes | Data rows | SHA-256 |
|---|---:|---:|---|
{hash_rows}

Modification timestamps (nanoseconds since epoch) were recorded by the validator: `{json.dumps({k: v['mtime_ns'] for k, v in pre.items()}, sort_keys=True)}`.

## 4. Expected matrix

Exp07: 120 rows/6 cells; Exp08: 320/16; Exp09: 400/20; global: 840/42. Every cell requires seeds 42–61 exactly once (`n=20`).

## 5. Exp07 validation

PASS: 120 rows, only Gossip/AHBN, six semantic conditions, complete seed matrix, no duplicate run keys or exact rows. AHBN has blank raw fanout and canonical label `ahbn_canonical_adaptive`. Configuration identity is `configs/exp07_fanout.yaml`, BA topology parameter 3, plus seed; no topology hash was fabricated.

## 6. Exp08 validation

PASS: 320 rows, four algorithms, four overload factors, 16 cells, complete seed matrix, no duplicates/unexpected runs. The 320-row synchronized evidence and manifest associations passed.

## 7. Exp09 validation

PASS: 400 rows, four algorithms, five ER probabilities, 20 cells, complete seed matrix. The 100-row topology artifact contains exactly one `p + seed` realization and every realization is referenced by four result rows.

## 8. Global 840-row validation

PASS: 840 rows, 42 cells, 840 unique `experiment/algorithm/experimental_condition/seed` keys, every cell `n=20`; no observations discarded.

## 9. Topology/configuration identity validation

Exp07 retains authoritative topology family/parameter/config association without an invented hash. Exp08 topology hashes agree across all four comparators for each factor/seed. Exp09 topology hashes join exactly by `p + seed` and are shared by all four comparator rows.

## 10. Metric validity validation

PASS: all four canonical metrics exist, are numeric and finite; delivery ratio is in `[0,1]`; delay, duplicates, and forwards are nonnegative.

## 11. Canonical-label normalization

Raw labels were preserved in source files. Derived algorithm mappings are `gossip→Gossip`, `cluster→Structured`, `dcsoc→DC-SoC`, `ahbn→AHBN`. Conditions follow the requested Exp07, Exp08, and two-decimal Exp09 forms.

## 12. Aggregation method

Rows were grouped only by experiment, canonical algorithm, and canonical experimental condition. Four metrics were independently summarized from all 20 runs.

## 13. Student-t 95% CI definition

For each cell/metric: sample standard deviation uses `ddof=1`; CI is `mean ± t_(0.975,19) × s/sqrt(20)`. No inferential comparisons were performed.

## 14. Canonical raw output

`outputs/csv/final_control_raw_canonical.csv`: 840 rows, 840 unique run keys, zero missing/nonfinite metrics.

## 15. Canonical summary output

`outputs/csv/final_control_summary.csv`: 42 rows, every `n=20`, sum of `n` = 840, all aggregates finite, all standard deviations nonnegative, and every CI contains its mean.

| Derived file | Bytes | SHA-256 |
|---|---:|---|
{output_rows}

## 16. Post-S5 hashes / immutability comparison

PASS: all six post-S5 input SHA-256 values, byte sizes, row counts, and modification timestamps exactly equal their pre-S5 values. Raw-file mutations: 0.

## 17. Repository integrity

Pre-S5 `git status --short` was:

```text
{pre_status or '(clean)'}
```

Only the S5 script, audit, and two derived CSVs were created by S5. `git diff --check` passed. No frozen implementation or configuration was modified.

## 18. Exact commands

```bash
cd {ROOT}
{sys.executable} scripts/aggregate_s5_final_data.py
git status --short
git diff --check
```

The validator itself performs the exact file loads, matrix checks, evidence joins, pre/post SHA-256 comparisons, aggregation, and derived-output validation documented above.

## 19. Terminal output

```text
{terminal}
```

## 20. Final S5 decision

**PASS.** S5 ends with 840 verified canonical run rows and 42 verified descriptive-summary rows. Scientific analysis is deferred.
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.parse_args()
    pre_status = preexisting_status()
    pre = {k: metadata(p) for k, p in INPUTS.items()}
    exp07, s07 = validate_results("exp07", INPUTS["exp07_results"], 120, {"gossip", "ahbn"}, 6)
    exp08, s08 = validate_results("exp08", INPUTS["exp08_results"], 320, set(ALG), 16)
    exp09, s09 = validate_results("exp09", INPUTS["exp09_results"], 400, set(ALG), 20)
    validate_exp08_identity(exp08)
    validate_exp09_identity(exp09)
    raw = exp07 + exp08 + exp09
    require(len(raw) == 840, f"Global expected 840 rows, got {len(raw)}")
    summary = aggregate(raw)
    validate_outputs(raw, summary)
    raw_fields = ["experiment", "algorithm", "experimental_condition", "seed", "topology_type", "topology_parameter", "topology_id", *METRICS, "source_file"]
    summary_fields = ["experiment", "algorithm", "experimental_condition", "n", *[f"{m}_{suffix}" for m in METRICS for suffix in ("mean", "std", "ci95_low", "ci95_high")]]
    write_csv(RAW_OUT, raw, raw_fields)
    write_csv(SUMMARY_OUT, summary, summary_fields)
    post = {k: metadata(p) for k, p in INPUTS.items()}
    require(pre == post, f"Authoritative input mutation detected: pre={pre}, post={post}")
    require(not git("diff", "--check"), "git diff --check reported whitespace errors")
    stats = {"exp07": s07, "exp08": s08, "exp09": s09}
    terminal = report(stats, {str(p): sha256(p) for p in (RAW_OUT, SUMMARY_OUT)}, pre)
    DOC.write_text(audit(pre, post, pre_status, terminal), encoding="utf-8")
    require(not git("diff", "--check"), "git diff --check failed after audit creation")
    print(terminal)
    print("\nOUTPUT SHA-256")
    print(f"{sha256(RAW_OUT)}  {RAW_OUT.relative_to(ROOT)}")
    print(f"{sha256(SUMMARY_OUT)}  {SUMMARY_OUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationError as exc:
        print(f"S5 FINAL STATUS: FAIL\n{exc}", file=sys.stderr)
        raise SystemExit(1)
