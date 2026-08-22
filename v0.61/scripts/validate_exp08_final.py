#!/usr/bin/env python3
"""Validate authoritative v0.61 Exp08 raw results and execution evidence."""
from __future__ import annotations
import argparse, csv, json, math
from collections import Counter
EXPECTED_STRATEGIES = {"gossip", "cluster", "dcsoc", "ahbn"}
EXPECTED_FACTORS = {1.0, 1.5, 2.0, 3.0}
EXPECTED_SEEDS = set(range(42, 62))

def load(path):
    with open(path, newline="", encoding="utf-8") as h: return list(csv.DictReader(h))

def main() -> int:
    ap = argparse.ArgumentParser(); ap.add_argument("--results", required=True); ap.add_argument("--evidence", required=True)
    a = ap.parse_args(); results, evidence = load(a.results), load(a.evidence)
    key = lambda r, factor_name: (r["strategy"], float(r[factor_name]), int(r["seed"]))
    result_keys = [key(r, "ch_overload_factor") for r in results]
    evidence_keys = [key(r, "overload_factor") for r in evidence]
    expected = {(s, f, seed) for s in EXPECTED_STRATEGIES for f in EXPECTED_FACTORS for seed in EXPECTED_SEEDS}
    metrics_ok = all(math.isfinite(float(r[m])) for r in results for m in ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards"))
    cells_ok = all({int(r["seed"]) for r in results if r["strategy"] == s and float(r["ch_overload_factor"]) == f} == EXPECTED_SEEDS for s in EXPECTED_STRATEGIES for f in EXPECTED_FACTORS)
    topology_ok = all(len({r["topology_identity"] for r in evidence if int(r["seed"]) == seed}) == 1 for seed in EXPECTED_SEEDS)
    drows = [r for r in evidence if r["strategy"] == "dcsoc"]
    dcsoc_ok = True
    roles = Counter()
    for r in drows:
        eligible = json.loads(r["dcsoc_eligible_overload_nodes"]); target = int(float(r["dcsoc_selected_overload_node"]))
        roles[r["dcsoc_selected_overload_role"]] += 1
        dcsoc_ok &= int(float(r["effective_message_source"])) == int(float(r["dcsoc_master"]))
        dcsoc_ok &= target in eligible and r["dcsoc_selected_overload_role"] in {"Master", "Core"}
        dcsoc_ok &= int(float(r["max_structural_obligations"])) > 3
    deterministic_targets = all(
        len({r["dcsoc_selected_overload_node"] for r in drows if int(r["seed"]) == seed}) == 1
        for seed in EXPECTED_SEEDS
    )
    checks = {
        "320 rows": len(results) == 320,
        "exact run grid": set(result_keys) == expected,
        "no duplicate runs": len(result_keys) == len(set(result_keys)),
        "16 cells x n=20": cells_ok,
        "metrics finite": metrics_ok,
        "evidence joins 1:1": len(evidence) == 320 and set(evidence_keys) == set(result_keys) and len(evidence_keys) == len(set(evidence_keys)),
        "topology mismatches=0": topology_ok,
        "DC-SoC 80 source/target/uncapped": len(drows) == 80 and dcsoc_ok,
        "DC-SoC target replay deterministic": deterministic_targets,
        "Tail targets=0": roles["Tail"] == 0,
    }
    print("v0.61 Exp08 final dataset validation")
    for name, passed in checks.items(): print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"missing_runs={len(expected - set(result_keys))} duplicate_runs={len(result_keys)-len(set(result_keys))} unexpected_runs={len(set(result_keys)-expected)}")
    print(f"dcsoc_master_targets={roles['Master']} dcsoc_core_targets={roles['Core']} dcsoc_tail_targets={roles['Tail']}")
    overall = all(checks.values()); print(f"FINAL DATASET RESULT: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1
if __name__ == "__main__": raise SystemExit(main())
