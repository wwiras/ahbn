#!/usr/bin/env python3
"""Analyze one validated ControlSim v0.63 Exp09 formal dataset."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t


STRATEGIES = ("gossip", "cluster", "dcsoc", "ahbn")
P_VALUES = (0.04, 0.06, 0.08, 0.10, 0.12)
SEEDS = set(range(42, 62))
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise ValueError(message)


def one(directory: Path, pattern: str) -> Path:
    paths = sorted(directory.glob(pattern))
    require(len(paths) == 1, f"{directory}: expected one {pattern}, found {len(paths)}")
    return paths[0]


def fanout_for(z: float) -> int:
    if z <= -0.25: return 2
    if z < 0.25: return 3
    if z < 0.90: return 4
    if z < 1.50: return 5
    return 6


def audit_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"experiment", "strategy", "seed", "topology_type", "topology_param", *METRICS}
    require(required <= set(frame.columns), f"missing result columns: {sorted(required - set(frame.columns))}")
    require(len(frame) == 400 and set(frame["experiment"]) == {"exp09"}, "formal row/experiment mismatch")
    require(set(frame["strategy"]) == set(STRATEGIES) and set(frame["topology_type"]) == {"er"}, "treatment mismatch")
    frame["density_p"] = frame["topology_param"].astype(float)
    require(set(frame["density_p"]) == set(P_VALUES), "density mismatch")
    require(not frame.duplicated(["strategy", "density_p", "seed"]).any(), "duplicate run key")
    for key, group in frame.groupby(["strategy", "density_p"]):
        require(len(group) == 20 and set(group["seed"].astype(int)) == SEEDS, f"incomplete seed matrix: {key}")
    values = frame[list(METRICS)].apply(pd.to_numeric, errors="coerce")
    require(values.notna().all().all() and values.map(math.isfinite).all().all(), "invalid primary metric")
    require(values["delivery_ratio"].between(0, 1).all(), "delivery outside [0,1]")
    require((values[["propagation_delay", "duplicates", "total_forwards"]] >= 0).all().all(), "negative metric")
    return frame


def audit_trace(path: Path) -> pd.DataFrame:
    trace = pd.read_csv(path)
    required = {"experiment", "strategy", "seed", "scenario_tag", "score", "weight", "mode", "fanout",
                "d_hat", "l_hat", "u_hat", "c_hat"}
    require(required <= set(trace.columns) and len(trace) > 0, "incomplete trace schema")
    require(set(trace["experiment"]) == {"exp09"} and set(trace["strategy"]) == {"ahbn"}, "trace treatment mismatch")
    require(set(trace["seed"].astype(int)) == SEEDS, "trace seed coverage mismatch")
    prefix = "edge_prob="
    require(trace["scenario_tag"].astype(str).str.startswith(prefix).all(), "invalid trace scenario tag")
    trace["density_p"] = trace["scenario_tag"].astype(str).str.removeprefix(prefix).astype(float)
    require(set(trace["density_p"]) == set(P_VALUES), "trace density coverage mismatch")
    numeric_cols = ["score", "weight", "fanout", "d_hat", "l_hat", "u_hat", "c_hat"]
    numeric = trace[numeric_cols].apply(pd.to_numeric, errors="coerce")
    require(numeric.notna().all().all() and numeric.map(math.isfinite).all().all(), "invalid trace value")
    actual = numeric["fanout"].astype(int)
    require((actual == numeric["score"].map(fanout_for)).all() and actual.isin(range(2, 7)).all(), "fanout mapping mismatch")
    modes = numeric["weight"].map(lambda value: "gossip" if value >= 0.5 else "cluster")
    require((trace["mode"] == modes).all(), "mode mapping mismatch")
    score_error = (numeric["score"] - (-numeric["d_hat"] + numeric["l_hat"] + numeric["u_hat"] + numeric["c_hat"])).abs()
    require((score_error <= 1e-12).all(), "canonical score mismatch")
    return trace


def aggregate(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        for p in P_VALUES:
            group = results[(results["strategy"] == strategy) & (results["density_p"] == p)]
            row: dict[str, object] = {"strategy": strategy, "density_p": p, "n": len(group)}
            critical = float(t.ppf(0.975, len(group) - 1))
            for metric in METRICS:
                values = group[metric].astype(float); mean = float(values.mean()); sd = float(values.std(ddof=1))
                half = critical * sd / math.sqrt(len(values))
                row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                            f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half})
            rows.append(row)
    return pd.DataFrame(rows)


def adaptive_summary(trace: pd.DataFrame, topology: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for p in P_VALUES:
        group = trace[trace["density_p"] == p]; fanout = group["fanout"].astype(int)
        row: dict[str, object] = {"density_p": p, "trace_rows": len(group)}
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
        topo = topology[topology["density_p"] == p].iloc[0]
        row.update({"mean_edge_count": float(topo["mean_edge_count"]), "mean_degree": float(topo["mean_degree"]),
                    "fanout_violations": 0, "mode_violations": 0})
        rows.append(row)
    return pd.DataFrame(rows)


def endpoint_changes(summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for strategy in STRATEGIES:
        low = summary[(summary["strategy"] == strategy) & (summary["density_p"] == P_VALUES[0])].iloc[0]
        high = summary[(summary["strategy"] == strategy) & (summary["density_p"] == P_VALUES[-1])].iloc[0]
        row: dict[str, object] = {"strategy": strategy}
        for metric in METRICS:
            a, b = float(low[f"{metric}_mean"]), float(high[f"{metric}_mean"])
            row.update({f"{metric}_p004": a, f"{metric}_p012": b, f"{metric}_delta": b - a,
                        f"{metric}_change_pct": 100 * (b - a) / a if a else math.nan})
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("formal_output", type=Path); args = parser.parse_args()
    root = args.formal_output.resolve(); require(root.is_dir(), f"missing formal output: {root}")
    log = root / "terminal.log"; require(log.is_file() and "Stage 4 exp09 ControlSim v0.63 formal" in log.read_text(), "not formal Exp09 output")
    results = audit_results(one(root / "outputs/csv", "exp09_results_*.csv"))
    trace = audit_trace(one(root / "outputs/csv", "exp09_adaptive_trace_*.csv"))
    topology_path = root / "exp09_v063_topology_summary.csv"; require(topology_path.is_file(), "missing topology audit summary")
    topology = pd.read_csv(topology_path); require(topology["mean_edge_count"].is_monotonic_increasing, "density evidence mismatch")
    summary = aggregate(results); adaptive = adaptive_summary(trace, topology); endpoints = endpoint_changes(summary)
    summary.to_csv(root / "exp09_v063_summary.csv", index=False, float_format="%.17g")
    adaptive.to_csv(root / "exp09_v063_ahbn_adaptive_summary.csv", index=False, float_format="%.17g")
    endpoints.to_csv(root / "exp09_v063_p004_to_p012.csv", index=False, float_format="%.17g")
    print(f"selected formal output directory: {root}")
    print("expected rows: 400\nactual rows: 400\nconditions: 20; n=20 each")
    print("DATASET AUDIT: PASS\nDENSITY EVIDENCE: PASS\nAHBN TRACE VALIDATION: PASS")
    print("\nPRIMARY RESULTS")
    for _, row in summary.iterrows():
        values = "  ".join(f"{m}={row[f'{m}_mean']:.6f} [{row[f'{m}_ci95_low']:.6f}, {row[f'{m}_ci95_high']:.6f}]" for m in METRICS)
        print(f"{row['strategy']:<7} p={row['density_p']:.2f} n={int(row['n'])}  {values}")
    print("\nAHBN ADAPTIVE BY DENSITY\n" + adaptive.to_string(index=False))
    print("\nP=0.04 TO P=0.12\n" + endpoints.to_string(index=False))
    print("TECHNICAL VALIDATION: PASS")


if __name__ == "__main__": main()
