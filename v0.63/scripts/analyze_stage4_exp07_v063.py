#!/usr/bin/env python3
"""Audit and summarize one completed ControlSim v0.63 Exp07 formal dataset."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t


METRICS = {
    "delivery": "delivery_ratio",
    "delay": "propagation_delay",
    "duplicates": "duplicates",
    "forwards": "total_forwards",
}
EXPECTED_SEEDS = set(range(42, 62))


class AuditError(RuntimeError):
    pass


def require(condition: bool, message: str) -> None:
    if not condition:
        raise AuditError(message)


def only_file(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    require(len(matches) == 1, f"{directory}: expected one {pattern}, found {len(matches)}: {matches}")
    return matches[0]


def expected_fanout(z: float) -> int:
    if z <= -0.25:
        return 2
    if z < 0.25:
        return 3
    if z < 0.90:
        return 4
    if z < 1.50:
        return 5
    return 6


def treatment(row: pd.Series) -> str:
    if row["strategy"] == "ahbn":
        require(pd.isna(row["fanout"]), f"AHBN row has fixed fanout: {row.to_dict()}")
        return "AHBN adaptive"
    require(row["strategy"] == "gossip", f"unexpected strategy: {row['strategy']!r}")
    fanout = int(row["fanout"])
    require(float(row["fanout"]) == fanout and fanout in range(2, 7), f"invalid Gossip fanout: {row['fanout']}")
    return f"Gossip f{fanout}"


def audit_results(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"experiment", "strategy", "seed", "fanout", *METRICS.values()}
    require(required <= set(frame.columns), f"{path}: missing columns {sorted(required - set(frame.columns))}")
    require(len(frame) == 120, f"{path}: expected 120 rows, got {len(frame)}")
    require(set(frame["experiment"]) == {"exp07"}, f"{path}: unexpected experiment labels")
    frame["treatment"] = frame.apply(treatment, axis=1)
    expected = {*(f"Gossip f{k}" for k in range(2, 7)), "AHBN adaptive"}
    require(set(frame["treatment"]) == expected, f"{path}: treatment set mismatch")
    require(not frame.duplicated(["treatment", "seed"]).any(), f"{path}: duplicate treatment/seed run")
    require(not frame.duplicated().any(), f"{path}: exact duplicate rows")
    for label, group in frame.groupby("treatment"):
        seeds = set(group["seed"].astype(int))
        require(len(group) == 20 and seeds == EXPECTED_SEEDS,
                f"{path}: {label} expected seeds 42-61 once, got {sorted(seeds)} ({len(group)} rows)")
    numeric = frame[list(METRICS.values())].apply(pd.to_numeric, errors="coerce")
    require(numeric.notna().all().all(), f"{path}: NaN/nonnumeric primary metric")
    require(numeric.map(math.isfinite).all().all(), f"{path}: nonfinite primary metric")
    require(numeric["delivery_ratio"].between(0, 1).all(), f"{path}: delivery_ratio outside [0,1]")
    require((numeric[["propagation_delay", "duplicates", "total_forwards"]] >= 0).all().all(),
            f"{path}: negative primary metric")
    return frame


def audit_trace(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"experiment", "strategy", "seed", "score", "weight", "mode", "fanout",
                "d_hat", "l_hat", "u_hat", "c_hat"}
    require(required <= set(frame.columns), f"{path}: missing trace columns {sorted(required - set(frame.columns))}")
    require(len(frame) > 0, f"{path}: empty AHBN trace")
    require(set(frame["experiment"]) == {"exp07"} and set(frame["strategy"]) == {"ahbn"},
            f"{path}: non-Exp07/AHBN trace row")
    require(set(frame["seed"].astype(int)) == EXPECTED_SEEDS, f"{path}: incomplete trace seed coverage")
    numeric_cols = ["score", "weight", "fanout", "d_hat", "l_hat", "u_hat", "c_hat"]
    numeric = frame[numeric_cols].apply(pd.to_numeric, errors="coerce")
    require(numeric.notna().all().all() and numeric.map(math.isfinite).all().all(), f"{path}: invalid trace numeric value")
    actual = numeric["fanout"].astype(int)
    require((actual == numeric["fanout"]).all() and actual.isin(range(2, 7)).all(), f"{path}: invalid fanout gear")
    expected = numeric["score"].map(expected_fanout)
    fanout_bad = actual != expected
    mode_expected = numeric["weight"].map(lambda value: "gossip" if value >= 0.5 else "cluster")
    mode_bad = frame["mode"] != mode_expected
    if fanout_bad.any():
        index = fanout_bad[fanout_bad].index[0]
        raise AuditError(f"{path}: row {index + 2}: z/fanout mismatch: z={numeric.at[index, 'score']}, fanout={actual.at[index]}, expected={expected.at[index]}")
    if mode_bad.any():
        index = mode_bad[mode_bad].index[0]
        raise AuditError(f"{path}: row {index + 2}: weight/mode mismatch: weight={numeric.at[index, 'weight']}, mode={frame.at[index, 'mode']}, expected={mode_expected.at[index]}")
    return frame


def summarize(results: pd.DataFrame) -> pd.DataFrame:
    order = [*(f"Gossip f{k}" for k in range(2, 7)), "AHBN adaptive"]
    rows = []
    for label in order:
        group = results[results["treatment"] == label]
        row: dict[str, object] = {"treatment": label, "strategy": "ahbn" if label.startswith("AHBN") else "gossip",
                                  "fanout": "" if label.startswith("AHBN") else int(label[-1]), "n": len(group)}
        critical = float(t.ppf(0.975, len(group) - 1))
        for short, column in METRICS.items():
            values = group[column].astype(float)
            mean, sd = float(values.mean()), float(values.std(ddof=1))
            half = critical * sd / math.sqrt(len(values))
            row.update({f"{short}_mean": mean, f"{short}_sd": sd,
                        f"{short}_ci95_low": mean - half, f"{short}_ci95_high": mean + half})
        rows.append(row)
    return pd.DataFrame(rows)


def trace_summary(trace: pd.DataFrame) -> dict[str, object]:
    fanout = trace["fanout"].astype(int)
    result: dict[str, object] = {"trace_rows": len(trace), "z_min": float(trace["score"].min()),
                                "z_mean": float(trace["score"].mean()), "z_max": float(trace["score"].max())}
    for name in ("d_hat", "l_hat", "u_hat", "c_hat"):
        result.update({f"{name}_min": float(trace[name].min()), f"{name}_mean": float(trace[name].mean()),
                       f"{name}_max": float(trace[name].max())})
    for gear in range(2, 7):
        result[f"fanout_{gear}_count"] = int((fanout == gear).sum())
        result[f"fanout_{gear}_proportion"] = float((fanout == gear).mean())
    for mode in ("gossip", "cluster"):
        result[f"{mode}_mode_count"] = int((trace["mode"] == mode).sum())
        result[f"{mode}_mode_proportion"] = float((trace["mode"] == mode).mean())
    result.update({"fanout_violations": 0, "mode_violations": 0})
    return result


def comparisons(summary: pd.DataFrame) -> pd.DataFrame:
    ahbn = summary[summary["treatment"] == "AHBN adaptive"].iloc[0]
    rows = []
    for gear in range(2, 7):
        gossip = summary[summary["treatment"] == f"Gossip f{gear}"].iloc[0]
        row: dict[str, object] = {"comparison": f"AHBN adaptive vs Gossip f{gear}", "gossip_fanout": gear}
        row["delivery_delta"] = ahbn["delivery_mean"] - gossip["delivery_mean"]
        row["delay_delta"] = ahbn["delay_mean"] - gossip["delay_mean"]
        row["delay_relative_pct"] = 100 * row["delay_delta"] / gossip["delay_mean"] if gossip["delay_mean"] else math.nan
        for short in ("duplicates", "forwards"):
            delta = ahbn[f"{short}_mean"] - gossip[f"{short}_mean"]
            row[f"{short}_delta"] = delta
            row[f"{short}_reduction_pct"] = 100 * (gossip[f"{short}_mean"] - ahbn[f"{short}_mean"]) / gossip[f"{short}_mean"] if gossip[f"{short}_mean"] else math.nan
        rows.append(row)
    return pd.DataFrame(rows)


def report(summary: pd.DataFrame, adaptive: dict[str, object], compare: pd.DataFrame) -> str:
    lines = ["EXP07 v0.63 FORMAL ANALYSIS", "", "PRIMARY RESULTS"]
    for _, row in summary.iterrows():
        lines.append(f"{row.treatment}: delivery={row.delivery_mean:.6f} [{row.delivery_ci95_low:.6f}, {row.delivery_ci95_high:.6f}]; "
                     f"delay={row.delay_mean:.6f} [{row.delay_ci95_low:.6f}, {row.delay_ci95_high:.6f}]; "
                     f"duplicates={row.duplicates_mean:.3f} [{row.duplicates_ci95_low:.3f}, {row.duplicates_ci95_high:.3f}]; "
                     f"forwards={row.forwards_mean:.3f} [{row.forwards_ci95_low:.3f}, {row.forwards_ci95_high:.3f}]")
    lines += ["", "AHBN ADAPTIVE STATE", json.dumps(adaptive, sort_keys=True), "", "AHBN VS FIXED GOSSIP"]
    for _, row in compare.iterrows():
        lines.append(f"f{int(row.gossip_fanout)}: delivery_delta={row.delivery_delta:+.6f}; delay_delta={row.delay_delta:+.6f} ({row.delay_relative_pct:+.2f}%); "
                     f"duplicates_delta={row.duplicates_delta:+.3f} (reduction={row.duplicates_reduction_pct:+.2f}%); "
                     f"forwards_delta={row.forwards_delta:+.3f} (reduction={row.forwards_reduction_pct:+.2f}%)")
    lines += ["", "DATASET AUDIT: PASS", "TECHNICAL VALIDATION: PASS"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formal_output", type=Path)
    args = parser.parse_args()
    root = args.formal_output.resolve()
    require(root.is_dir(), f"formal output directory does not exist: {root}")
    terminal = root / "terminal.log"
    require(terminal.is_file(), f"missing terminal log: {terminal}")
    log_text = terminal.read_text(encoding="utf-8")
    require("Stage 4 exp07 ControlSim v0.63 formal" in log_text, f"{root}: not a formal Exp07 v0.63 output")
    require("TECHNICAL VALIDATION: PASS" in log_text and "EXIT CODE: 0" in log_text, f"{root}: incomplete/failed formal run")
    csv_dir = root / "outputs" / "csv"
    results_path = only_file(csv_dir, "exp07_results_*.csv")
    trace_path = only_file(csv_dir, "exp07_adaptive_trace_*.csv")
    results, trace = audit_results(results_path), audit_trace(trace_path)
    summary, adaptive = summarize(results), trace_summary(trace)
    compare = comparisons(summary)
    summary.to_csv(root / "exp07_v063_summary.csv", index=False, float_format="%.17g")
    pd.DataFrame([adaptive]).to_csv(root / "exp07_v063_ahbn_adaptive_summary.csv", index=False, float_format="%.17g")
    compare.to_csv(root / "exp07_v063_ahbn_vs_gossip.csv", index=False, float_format="%.17g")
    summary[["treatment", "strategy", "fanout", *[f"{m}_mean" for m in METRICS]]].to_csv(
        root / "exp07_v063_figure_data.csv", index=False, float_format="%.17g")
    text = report(summary, adaptive, compare)
    (root / "exp07_v063_analysis.txt").write_text(text, encoding="utf-8")
    print(f"selected formal output directory: {root}")
    print(f"timestamp: {root.name.rsplit('-', 1)[-1]}")
    print("result file count: 1")
    print("trace file count: 1")
    print("config: /Users/wwiras/Documents/src/AHBNProj/ahbn/v0.63/configs/exp07_fanout.yaml")
    print(f"results: {results_path}")
    print(f"trace: {trace_path}")
    print("expected rows: 120")
    print(f"actual rows: {len(results)}")
    print("treatment counts: " + json.dumps(results["treatment"].value_counts().sort_index().to_dict()))
    print(text, end="")
    print(f"Saved {root / 'exp07_v063_summary.csv'}")
    print(f"Saved {root / 'exp07_v063_ahbn_adaptive_summary.csv'}")
    print(f"Saved {root / 'exp07_v063_ahbn_vs_gossip.csv'}")
    print(f"Saved {root / 'exp07_v063_figure_data.csv'}")
    print(f"Saved {root / 'exp07_v063_analysis.txt'}")


if __name__ == "__main__":
    main()
