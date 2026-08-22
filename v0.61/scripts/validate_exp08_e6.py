#!/usr/bin/env python3
"""Validate a newly produced v0.61 Exp08 AHBN trace."""
from __future__ import annotations
import argparse, csv, math
from collections import Counter
EXPECTED = {(f, s) for f in (1.0, 1.5, 2.0, 3.0) for s in range(42, 62)}

def factor(tag: str) -> float: return float(tag.split("=", 1)[1])
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True)
    ap.add_argument("--trace", required=True)
    args = ap.parse_args()
    with open(args.results, newline="", encoding="utf-8") as h:
        result_runs = {(float(r["ch_overload_factor"]), int(r["seed"])) for r in csv.DictReader(h) if r["strategy"] == "ahbn"}
    with open(args.trace, newline="", encoding="utf-8") as h: rows = list(csv.DictReader(h))
    trace_runs = {(factor(r["scenario_tag"]), int(r["seed"])) for r in rows}
    fanouts = [int(r["fanout"]) for r in rows]
    modes = {r["mode"] for r in rows}
    finite_fields = ("time", "d_hat", "l_hat", "u_hat", "c_hat", "score", "weight")
    finite = all(all(math.isfinite(float(r[x])) for x in finite_fields) for r in rows)
    fanout_changes = sum(r["fanout_changed"] == "True" for r in rows)
    mode_changes = sum(r["mode_switched"] == "True" for r in rows)
    checks = {
        "80 AHBN result runs": result_runs == EXPECTED,
        "all run contexts represented": trace_runs == EXPECTED,
        "trace nonempty": bool(rows),
        "finite controller values": finite,
        "fanout within [2,4]": bool(fanouts) and min(fanouts) >= 2 and max(fanouts) <= 4,
        "valid modes": modes <= {"gossip", "cluster"} and bool(modes),
    }
    print("E6 — v0.61 Exp08 AHBN trace validation")
    for name, passed in checks.items(): print(f"{'PASS' if passed else 'FAIL'}  {name}")
    print(f"trace_rows={len(rows)}")
    print(f"fanout_min={min(fanouts) if fanouts else 'N/A'} fanout_max={max(fanouts) if fanouts else 'N/A'}")
    print(f"fanout_transitions={fanout_changes} mode_transitions={mode_changes}")
    print(f"modes_observed={sorted(modes)}")
    overall = all(checks.values())
    print(f"E6 RESULT: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1
if __name__ == "__main__": raise SystemExit(main())
