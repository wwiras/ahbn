"""Validate exact official Exp08/Exp09 result and AHBN trace files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


STRATEGIES = {"gossip", "cluster", "dcsoc", "ahbn"}
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def validate_results(
    experiment: str,
    path: Path,
    condition_field: str,
    conditions: list[str],
    seeds: set[int],
) -> bool:
    rows = read_rows(path)
    identities = [(r["strategy"], r[condition_field], int(r["seed"])) for r in rows]
    cells = Counter((r["strategy"], r[condition_field]) for r in rows)
    checks = {
        "experiment": {r["experiment"] for r in rows} == {experiment},
        "row_count": len(rows) == len(STRATEGIES) * len(conditions) * len(seeds),
        "strategies": {r["strategy"] for r in rows} == STRATEGIES,
        "conditions": {r[condition_field] for r in rows} == set(conditions),
        "unique_identities": len(identities) == len(set(identities)),
        "complete_cells": set(cells) == {(s, c) for s in STRATEGIES for c in conditions}
        and set(cells.values()) == {len(seeds)},
        "seed_coverage": all(
            {int(r["seed"]) for r in rows if r["strategy"] == s and r[condition_field] == c}
            == seeds
            for s in STRATEGIES
            for c in conditions
        ),
        "finite_metrics": all(
            math.isfinite(float(r[m])) for r in rows for m in METRICS
        ),
        "delivery_range": all(0.0 <= float(r["delivery_ratio"]) <= 1.0 for r in rows),
        "nonnegative_counts": all(
            float(r[m]) >= 0.0 for r in rows for m in ("duplicates", "total_forwards")
        ),
        "fanout_metadata_unset": all(r["fanout"] == "" for r in rows),
    }
    print(f"{experiment.upper()} RESULTS: rows={len(rows)} cells={len(cells)}")
    print(f"{experiment.upper()} STRATEGY_COUNTS: {dict(sorted(Counter(r['strategy'] for r in rows).items()))}")
    print(f"{experiment.upper()} CONDITION_COUNTS: {dict(sorted(Counter(r[condition_field] for r in rows).items()))}")
    for name, passed in checks.items():
        print(f"{experiment.upper()} results {name}: {'PASS' if passed else 'FAIL'}")
    return all(checks.values())


def validate_trace(
    experiment: str,
    path: Path,
    scenarios: set[str],
    seeds: set[int],
) -> bool:
    rows = read_rows(path)
    numeric = ("time", "d_hat", "l_hat", "u_hat", "c_hat", "score", "weight", "fanout")
    checks = {
        "nonempty": bool(rows),
        "experiment": {r["experiment"] for r in rows} == {experiment},
        "ahbn_only": {r["strategy"] for r in rows} == {"ahbn"},
        "scenarios": {r["scenario_tag"] for r in rows} == scenarios,
        "seeds": {int(r["seed"]) for r in rows} == seeds,
        "all_run_cells": {
            (r["scenario_tag"], int(r["seed"])) for r in rows
        } == {(scenario, seed) for scenario in scenarios for seed in seeds},
        "finite_controller_fields": all(
            r[field] != "" and math.isfinite(float(r[field]))
            for r in rows
            for field in numeric
        ),
        "adaptive_fanout_bounds": all(2 <= int(r["fanout"]) <= 4 for r in rows),
    }
    print(f"{experiment.upper()} TRACE: rows={len(rows)}")
    for name, passed in checks.items():
        print(f"{experiment.upper()} trace {name}: {'PASS' if passed else 'FAIL'}")
    return all(checks.values())


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp08-results", type=Path, required=True)
    parser.add_argument("--exp08-trace", type=Path, required=True)
    parser.add_argument("--exp09-results", type=Path, required=True)
    parser.add_argument("--exp09-trace", type=Path, required=True)
    args = parser.parse_args()

    exp08_conditions = ["1.0", "1.5", "2.0", "3.0"]
    exp09_conditions = ["0.04", "0.06", "0.08", "0.1", "0.12"]
    exp08_seeds = set(range(42, 62))
    exp09_seeds = set(range(42, 72))
    checks = [
        validate_results("exp08", args.exp08_results, "ch_overload_factor", exp08_conditions, exp08_seeds),
        validate_trace("exp08", args.exp08_trace, {f"ch_overload_factor={v}" for v in exp08_conditions}, exp08_seeds),
        validate_results("exp09", args.exp09_results, "topology_param", exp09_conditions, exp09_seeds),
        validate_trace("exp09", args.exp09_trace, {f"edge_prob={v}" for v in exp09_conditions}, exp09_seeds),
    ]
    passed = all(checks)
    print(f"OFFICIAL EXP08/EXP09 DATA-INTEGRITY GATE: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
