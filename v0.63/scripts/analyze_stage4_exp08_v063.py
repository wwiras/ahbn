#!/usr/bin/env python3
"""Audit and analyze one completed ControlSim v0.63 Exp08 formal dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t


STRATEGIES = ("gossip", "cluster", "dcsoc", "ahbn")
OVERLOADS = (1.0, 1.5, 2.0, 3.0)
SEEDS = set(range(42, 62))
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def one(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    require(len(paths) == 1, f"{directory}: expected one {pattern}, found {len(paths)}: {paths}")
    return paths[0]


def fanout_for(z: float) -> int:
    if z <= -0.25: return 2
    if z < 0.25: return 3
    if z < 0.90: return 4
    if z < 1.50: return 5
    return 6


def audit_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"experiment", "strategy", "seed", "ch_overload_factor", *METRICS}
    require(required <= set(frame.columns), f"{path}: missing columns {sorted(required - set(frame.columns))}")
    require(len(frame) == 320, f"{path}: expected 320 rows, got {len(frame)}")
    require(set(frame["experiment"]) == {"exp08"}, f"{path}: experiment mismatch")
    require(set(frame["strategy"]) == set(STRATEGIES), f"{path}: strategy mismatch")
    frame["ch_overload_factor"] = frame["ch_overload_factor"].astype(float)
    require(set(frame["ch_overload_factor"]) == set(OVERLOADS), f"{path}: overload mismatch")
    require(not frame.duplicated(["strategy", "ch_overload_factor", "seed"]).any(), f"{path}: duplicate run key")
    for key, group in frame.groupby(["strategy", "ch_overload_factor"]):
        require(len(group) == 20 and set(group["seed"].astype(int)) == SEEDS,
                f"{path}: incomplete seed matrix for {key}")
    values = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    require(values.notna().all().all() and values.map(math.isfinite).all().all(), f"{path}: invalid metric")
    require(values["delivery_ratio"].between(0, 1).all(), f"{path}: delivery outside [0,1]")
    require((values[["propagation_delay", "duplicates", "total_forwards"]] >= 0).all().all(), f"{path}: negative metric")
    return frame


def audit_evidence(path: Path, results: pd.DataFrame) -> None:
    evidence = pd.read_csv(path)
    keys = ["strategy", "overload_factor", "seed"]
    require(len(evidence) == 320 and not evidence.duplicated(keys).any(), f"{path}: invalid evidence keys")
    a = set(zip(results.strategy, results.ch_overload_factor.astype(float), results.seed.astype(int)))
    b = set(zip(evidence.strategy, evidence.overload_factor.astype(float), evidence.seed.astype(int)))
    require(a == b, f"{path}: result/evidence join mismatch")
    identities = evidence.groupby(["overload_factor", "seed"])["topology_identity"].nunique()
    require((identities == 1).all(), f"{path}: paired topology mismatch")
    drows = evidence[evidence.strategy == "dcsoc"]
    for index, row in drows.iterrows():
        eligible = json.loads(row.dcsoc_eligible_overload_nodes)
        target = int(row.dcsoc_selected_overload_node)
        require(int(row.effective_message_source) == int(row.dcsoc_master), f"{path}:{index + 2}: DC-SoC source mismatch")
        require(target in eligible and row.dcsoc_selected_overload_role in {"Master", "Core"},
                f"{path}:{index + 2}: DC-SoC overload target mismatch")
        require(int(row.max_structural_obligations) > 3, f"{path}:{index + 2}: DC-SoC obligations capped")


def audit_trace(path: Path) -> pd.DataFrame:
    trace = pd.read_csv(path)
    required = {"experiment", "strategy", "seed", "scenario_tag", "score", "weight", "mode", "fanout",
                "d_hat", "l_hat", "u_hat", "c_hat"}
    require(required <= set(trace.columns) and len(trace) > 0, f"{path}: incomplete trace schema")
    require(set(trace.experiment) == {"exp08"} and set(trace.strategy) == {"ahbn"}, f"{path}: trace treatment mismatch")
    require(set(trace.seed.astype(int)) == SEEDS, f"{path}: trace seed coverage mismatch")
    prefix = "ch_overload_factor="
    require(trace.scenario_tag.astype(str).str.startswith(prefix).all(), f"{path}: invalid scenario tags")
    trace["overload_factor"] = trace.scenario_tag.astype(str).str.removeprefix(prefix).astype(float)
    require(set(trace.overload_factor) == set(OVERLOADS), f"{path}: trace overload coverage mismatch")
    numeric_cols = ["score", "weight", "fanout", "d_hat", "l_hat", "u_hat", "c_hat"]
    numeric = trace[numeric_cols].apply(pd.to_numeric, errors="coerce")
    require(numeric.notna().all().all() and numeric.map(math.isfinite).all().all(), f"{path}: invalid trace value")
    actual = numeric.fanout.astype(int)
    expected = numeric.score.map(fanout_for)
    require((actual == numeric.fanout).all() and actual.isin(range(2, 7)).all(), f"{path}: invalid fanout gear")
    require((actual == expected).all(), f"{path}: score/fanout mismatch")
    modes = numeric.weight.map(lambda value: "gossip" if value >= 0.5 else "cluster")
    require((trace["mode"] == modes).all(), f"{path}: weight/mode mismatch")
    return trace


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        for overload in OVERLOADS:
            group = results[(results.strategy == strategy) & (results.ch_overload_factor == overload)]
            row: dict[str, object] = {"strategy": strategy, "ch_overload_factor": overload, "n": len(group)}
            critical = float(t.ppf(0.975, len(group) - 1))
            for metric in METRICS:
                values = group[metric].astype(float)
                mean, sd = float(values.mean()), float(values.std(ddof=1))
                half = critical * sd / math.sqrt(len(values))
                row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                            f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half})
            rows.append(row)
    return pd.DataFrame(rows)


def adaptive_summary(trace: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for overload in OVERLOADS:
        group = trace[trace.overload_factor == overload]
        fanout = group.fanout.astype(int)
        row: dict[str, object] = {"ch_overload_factor": overload, "trace_rows": len(group)}
        for name in ("score", "d_hat", "l_hat", "u_hat", "c_hat"):
            label = "z" if name == "score" else name
            row.update({f"{label}_min": float(group[name].min()), f"{label}_mean": float(group[name].mean()),
                        f"{label}_max": float(group[name].max())})
        for gear in range(2, 7):
            row[f"fanout_{gear}_count"] = int((fanout == gear).sum())
            row[f"fanout_{gear}_proportion"] = float((fanout == gear).mean())
        for mode in ("gossip", "cluster"):
            row[f"{mode}_mode_count"] = int((group["mode"] == mode).sum())
            row[f"{mode}_mode_proportion"] = float((group["mode"] == mode).mean())
        row.update({"fanout_violations": 0, "mode_violations": 0})
        rows.append(row)
    return pd.DataFrame(rows)


def changes(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        low = summary[(summary.strategy == strategy) & (summary.ch_overload_factor == 1.0)].iloc[0]
        high = summary[(summary.strategy == strategy) & (summary.ch_overload_factor == 3.0)].iloc[0]
        row: dict[str, object] = {"strategy": strategy}
        for metric in METRICS:
            a, b = float(low[f"{metric}_mean"]), float(high[f"{metric}_mean"])
            row.update({f"{metric}_baseline": a, f"{metric}_highest": b, f"{metric}_delta": b - a,
                        f"{metric}_change_pct": 100 * (b - a) / a if a else math.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("formal_output", type=Path); args = parser.parse_args()
    root = args.formal_output.resolve(); require(root.is_dir(), f"missing formal output: {root}")
    log = root / "terminal.log"; require(log.is_file() and "Stage 4 exp08 ControlSim v0.63 formal" in log.read_text(), f"not a formal Exp08 output: {root}")
    csv = root / "outputs" / "csv"
    results_path = one(csv, "exp08_results_*.csv")
    evidence_path = one(csv, "exp08_execution_evidence_*.csv")
    trace_path = one(csv, "exp08_ahbn_adaptive_trace_*.csv")
    results = audit_results(results_path); audit_evidence(evidence_path, results); trace = audit_trace(trace_path)
    summary, adaptive = aggregate(results), adaptive_summary(trace); trend = changes(summary)
    summary.to_csv(root / "exp08_v063_summary.csv", index=False, float_format="%.17g")
    adaptive.to_csv(root / "exp08_v063_ahbn_adaptive_summary.csv", index=False, float_format="%.17g")
    trend.to_csv(root / "exp08_v063_baseline_to_highest.csv", index=False, float_format="%.17g")
    print(f"selected formal output directory: {root}")
    print(f"results: {results_path}\nevidence: {evidence_path}\ntrace: {trace_path}")
    print(f"expected rows: 320\nactual rows: {len(results)}\nconditions: 16; n=20 each")
    print("DATASET AUDIT: PASS\nEVIDENCE/OVERLOAD CONTRACT: PASS\nAHBN TRACE VALIDATION: PASS")
    print("\nPRIMARY RESULTS")
    for _, row in summary.iterrows():
        values = "  ".join(f"{m}={row[f'{m}_mean']:.6f} [{row[f'{m}_ci95_low']:.6f}, {row[f'{m}_ci95_high']:.6f}]" for m in METRICS)
        print(f"{row.strategy:<7} overload={row.ch_overload_factor:.1f} n={int(row.n)}  {values}")
    print("\nAHBN ADAPTIVE BY OVERLOAD")
    print(adaptive.to_string(index=False))
    print("\nBASELINE TO HIGHEST OVERLOAD")
    print(trend.to_string(index=False))
    print(f"Saved {root / 'exp08_v063_summary.csv'}")
    print(f"Saved {root / 'exp08_v063_ahbn_adaptive_summary.csv'}")
    print(f"Saved {root / 'exp08_v063_baseline_to_highest.csv'}")
    print("TECHNICAL VALIDATION: PASS")


if __name__ == "__main__": main()
