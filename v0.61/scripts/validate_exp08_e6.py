#!/usr/bin/env python3
"""Validate Exp08 E6 using only the frozen E5 AHBN CSV evidence."""

from __future__ import annotations

import csv
import math
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TRACE_PATH = ROOT / "outputs/csv/exp08_ahbn_adaptive_trace_20260820_115817.csv"
RESULTS_PATH = ROOT / "outputs/csv/exp08_ahbn_results_20260820_115817.csv"

EXPECTED_FACTORS = (1.0, 1.5, 2.0, 3.0)
EXPECTED_SEEDS = tuple(range(42, 62))
EXPECTED_RUNS = {(factor, seed) for factor in EXPECTED_FACTORS for seed in EXPECTED_SEEDS}
VALID_MODES = {"gossip", "cluster"}
MIN_FANOUT = 2
MAX_FANOUT = 4

FLOAT_FIELDS = (
    "time", "duplicate_obs", "latency_obs", "utilization_obs", "churn_obs",
    "d_hat", "l_hat", "u_hat", "c_hat", "score", "weight",
    "duplicate_ratio_raw", "capacity_score", "processing_delay",
)
REQUIRED_FLOAT_FIELDS = (
    "time", "duplicate_obs", "latency_obs", "utilization_obs",
    "d_hat", "l_hat", "u_hat", "c_hat", "score", "weight",
    "duplicate_ratio_raw", "capacity_score", "processing_delay",
)
BOUNDED_FIELDS = (
    "duplicate_obs", "latency_obs", "utilization_obs", "churn_obs",
    "d_hat", "l_hat", "u_hat", "c_hat", "weight", "duplicate_ratio_raw",
)
INTEGER_FIELDS = (
    "seed", "node_id", "fanout", "received_new", "received_duplicate", "forwarded",
)
REQUIRED_TRACE_COLUMNS = {
    "experiment", "strategy", "seed", "scenario_tag", "time", "node_id",
    "message_id", "event_type", "duplicate_obs", "latency_obs",
    "utilization_obs", "churn_obs", "d_hat", "l_hat", "u_hat", "c_hat",
    "score", "weight", "mode", "fanout", "mode_switched",
    "fanout_changed", "duplicate_ratio_raw", "resource_class",
    "capacity_score", "processing_delay", "received_new",
    "received_duplicate", "forwarded",
}


def fail(message: str, errors: list[str]) -> None:
    errors.append(message)


def parse_bool(value: str) -> bool:
    if value == "True":
        return True
    if value == "False":
        return False
    raise ValueError(f"invalid boolean {value!r}")


def factor_from_tag(tag: str) -> float:
    prefix = "ch_overload_factor="
    if not tag.startswith(prefix):
        raise ValueError(f"invalid scenario_tag {tag!r}")
    return float(tag[len(prefix):])


def describe(values: list[float]) -> str:
    return (
        f"n={len(values):5d} mean={statistics.fmean(values):.6f} "
        f"median={statistics.median(values):.6f} "
        f"min={min(values):.6f} max={max(values):.6f}"
    )


def pct(count: int, total: int) -> str:
    return f"{100.0 * count / total:.2f}%" if total else "0.00%"


def load_results(errors: list[str]) -> set[tuple[float, int]]:
    if not RESULTS_PATH.is_file():
        fail(f"results file missing: {RESULTS_PATH}", errors)
        return set()
    identities: list[tuple[float, int]] = []
    with RESULTS_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"experiment", "strategy", "seed", "ch_overload_factor"}
        if not required.issubset(reader.fieldnames or []):
            fail("results CSV is missing run-identity columns", errors)
            return set()
        for line, row in enumerate(reader, 2):
            try:
                if row["experiment"] != "exp08" or row["strategy"] != "ahbn":
                    raise ValueError("non-Exp08/AHBN result row")
                identities.append((float(row["ch_overload_factor"]), int(row["seed"])))
            except (TypeError, ValueError) as exc:
                fail(f"results row {line}: {exc}", errors)
    counts = Counter(identities)
    if set(counts) != EXPECTED_RUNS or any(value != 1 for value in counts.values()):
        fail("results do not contain exactly one row for each of the 80 expected runs", errors)
    return set(counts)


def load_trace(errors: list[str]) -> list[dict]:
    if not TRACE_PATH.is_file():
        fail(f"trace file missing: {TRACE_PATH}", errors)
        return []
    parsed: list[dict] = []
    with TRACE_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_TRACE_COLUMNS - set(reader.fieldnames or [])
        if missing:
            fail(f"trace CSV missing columns: {sorted(missing)}", errors)
            return []
        for line, source in enumerate(reader, 2):
            try:
                row = dict(source)
                for field in FLOAT_FIELDS:
                    row[field] = None if source[field] == "" else float(source[field])
                for field in INTEGER_FIELDS:
                    row[field] = int(source[field])
                row["mode_switched"] = parse_bool(source["mode_switched"])
                row["fanout_changed"] = parse_bool(source["fanout_changed"])
                row["factor"] = factor_from_tag(source["scenario_tag"])
                row["line"] = line
                parsed.append(row)
            except (TypeError, ValueError) as exc:
                fail(f"trace row {line}: {exc}", errors)
    if not parsed:
        fail("trace is empty", errors)
    return parsed


def validate_rows(rows: list[dict], result_runs: set[tuple[float, int]], errors: list[str]) -> None:
    trace_runs = {(row["factor"], row["seed"]) for row in rows}
    if trace_runs != EXPECTED_RUNS:
        fail("trace does not represent exactly the expected 80 factor/seed runs", errors)
    if result_runs and trace_runs != result_runs:
        fail("trace run identities do not match the frozen results", errors)

    for row in rows:
        line = row["line"]
        if row["experiment"] != "exp08" or row["strategy"] != "ahbn":
            fail(f"trace row {line}: unexpected experiment/strategy", errors)
        if row["factor"] not in EXPECTED_FACTORS or row["seed"] not in EXPECTED_SEEDS:
            fail(f"trace row {line}: unexpected factor or seed", errors)
        if row["mode"] not in VALID_MODES:
            fail(f"trace row {line}: invalid mode {row['mode']!r}", errors)
        if not MIN_FANOUT <= row["fanout"] <= MAX_FANOUT:
            fail(f"trace row {line}: fanout outside [{MIN_FANOUT}, {MAX_FANOUT}]", errors)
        if not 0 <= row["node_id"] < 100 or row["time"] is None or row["time"] < 0:
            fail(f"trace row {line}: invalid node/time", errors)
        if not row["event_type"]:
            fail(f"trace row {line}: missing event type", errors)
        for field in REQUIRED_FLOAT_FIELDS:
            value = row[field]
            if value is None or not math.isfinite(value):
                fail(f"trace row {line}: {field} is missing/non-finite", errors)
        if row["churn_obs"] is not None and not math.isfinite(row["churn_obs"]):
            fail(f"trace row {line}: churn_obs is non-finite", errors)
        for field in BOUNDED_FIELDS:
            value = row[field]
            if value is not None and not 0.0 <= value <= 1.0:
                fail(f"trace row {line}: {field} outside [0, 1]", errors)
        if row["capacity_score"] <= 0 or row["processing_delay"] < 0:
            fail(f"trace row {line}: invalid resource diagnostics", errors)
        if any(row[field] < 0 for field in ("received_new", "received_duplicate", "forwarded")):
            fail(f"trace row {line}: negative cumulative counter", errors)


def controller_consistency(rows: list[dict], errors: list[str]) -> int:
    mismatches = 0
    for row in rows:
        expected_score = -row["d_hat"] + row["l_hat"] - row["u_hat"] + row["c_hat"]
        expected_weight = 1.0 / (1.0 + math.exp(-expected_score))
        expected_mode = "gossip" if expected_weight >= 0.5 else "cluster"
        expected_fanout = round(MIN_FANOUT + expected_weight * (MAX_FANOUT - MIN_FANOUT))
        consistent = (
            math.isclose(row["score"], expected_score, rel_tol=0.0, abs_tol=1e-12)
            and math.isclose(row["weight"], expected_weight, rel_tol=0.0, abs_tol=1e-12)
            and row["mode"] == expected_mode
            and row["fanout"] == expected_fanout
        )
        if not consistent:
            mismatches += 1
            if mismatches <= 5:
                fail(f"trace row {row['line']}: controller decision mismatch", errors)
    return mismatches


def main() -> int:
    errors: list[str] = []
    result_runs = load_results(errors)
    rows = load_trace(errors)
    if rows:
        validate_rows(rows, result_runs, errors)

    by_factor: dict[float, list[dict]] = defaultdict(list)
    by_run: dict[tuple[float, int], list[dict]] = defaultdict(list)
    for row in rows:
        by_factor[row["factor"]].append(row)
        by_run[(row["factor"], row["seed"])].append(row)

    latency = {factor: [row["latency_obs"] for row in by_factor[factor]] for factor in EXPECTED_FACTORS}
    utilization = {factor: [row["utilization_obs"] for row in by_factor[factor]] for factor in EXPECTED_FACTORS}
    observation_pass = bool(rows)
    for name, grouped in (("latency", latency), ("utilization", utilization)):
        flat = [value for values in grouped.values() for value in values]
        if not flat or len(set(flat)) <= 1:
            fail(f"{name} observation stream is missing or constant", errors)
            observation_pass = False
        if len({round(statistics.fmean(values), 12) for values in grouped.values() if values}) <= 1:
            fail(f"{name} observations show no factor-associated change", errors)
            observation_pass = False

    mode_transition_counts = {
        run: sum(row["mode_switched"] for row in run_rows) for run, run_rows in by_run.items()
    }
    fanout_transition_counts = {
        run: sum(row["fanout_changed"] for row in run_rows) for run, run_rows in by_run.items()
    }
    mismatches = controller_consistency(rows, errors) if rows else 0

    print("E6 — Exp08 AHBN Adaptive Trace Validation")
    print()
    print("Trace:")
    print(f"  file: {TRACE_PATH.relative_to(ROOT)}")
    print(f"  rows: {len(rows):,}")
    print(f"  runs: {len(by_run)}")
    print(f"  seeds: {min((row['seed'] for row in rows), default='N/A')}–{max((row['seed'] for row in rows), default='N/A')}")
    print(f"  overload factors: {sorted(by_factor)}")
    print()
    print(f"Trace integrity: {'PASS' if not errors else 'FAIL'}")
    print()
    print("Utilization/latency response:")
    for factor in EXPECTED_FACTORS:
        print(f"  factor={factor:.1f}")
        print(f"    latency:    {describe(latency[factor])}")
        print(f"    utilization:{describe(utilization[factor])}")
    print(f"  Assessment: {'PASS' if observation_pass else 'FAIL'} — non-constant runtime observations with factor-associated variation")
    print()
    print("Mode behaviour:")
    for factor in EXPECTED_FACTORS:
        counts = Counter(row["mode"] for row in by_factor[factor])
        total = len(by_factor[factor])
        transitions = sum(mode_transition_counts.get((factor, seed), 0) for seed in EXPECTED_SEEDS)
        print(
            f"  factor={factor:.1f}: "
            f"gossip={counts['gossip']:,} ({pct(counts['gossip'], total)}), "
            f"cluster={counts['cluster']:,} ({pct(counts['cluster'], total)}), "
            f"transitions={transitions:,}"
        )
    mode_total = sum(mode_transition_counts.values())
    mode_runs = sum(value > 0 for value in mode_transition_counts.values())
    print(f"  total transitions: {mode_total:,}")
    print(f"  runs with transitions: {mode_runs}; runs with zero transitions: {len(by_run) - mode_runs}")
    print("  Assessment: PASS" if rows and all(row["mode"] in VALID_MODES for row in rows) else "  Assessment: FAIL")
    print()
    print("Fanout behaviour:")
    for factor in EXPECTED_FACTORS:
        counts = Counter(row["fanout"] for row in by_factor[factor])
        total = len(by_factor[factor])
        transitions = sum(fanout_transition_counts.get((factor, seed), 0) for seed in EXPECTED_SEEDS)
        details = ", ".join(f"fanout={value}: {counts[value]:,} ({pct(counts[value], total)})" for value in range(MIN_FANOUT, MAX_FANOUT + 1))
        print(f"  factor={factor:.1f}: {details}, transitions={transitions:,}")
    fanout_total = sum(fanout_transition_counts.values())
    fanout_runs = sum(value > 0 for value in fanout_transition_counts.values())
    observed = [row["fanout"] for row in rows]
    print(f"  observed range: {min(observed, default='N/A')}–{max(observed, default='N/A')} (frozen bounds: {MIN_FANOUT}–{MAX_FANOUT})")
    print(f"  total transitions: {fanout_total:,}")
    print(f"  runs with transitions: {fanout_runs}; runs with zero transitions: {len(by_run) - fanout_runs}")
    fanout_ok = bool(rows) and all(MIN_FANOUT <= value <= MAX_FANOUT for value in observed)
    print(f"  Assessment: {'PASS' if fanout_ok else 'FAIL'}")
    if observed and len(set(observed)) == 1:
        print(
            "  SCIENTIFIC OBSERVATION: fanout remained at "
            f"{observed[0]} throughout the trace; this is valid within the frozen bounds."
        )
    print()
    print("Controller consistency check:")
    print(f"  recorded decision mismatches: {mismatches:,}")
    print(f"  Assessment: {'PASS' if mismatches == 0 and rows else 'FAIL'}")
    print()
    if errors:
        print("Validation problems:")
        for message in errors[:20]:
            print(f"  - {message}")
        if len(errors) > 20:
            print(f"  - ... {len(errors) - 20} additional problem(s)")
        print()
    overall = bool(rows) and not errors and observation_pass and fanout_ok and mismatches == 0
    print(f"Overall E6: {'PASS' if overall else 'FAIL'}")
    return 0 if overall else 1


if __name__ == "__main__":
    sys.exit(main())
