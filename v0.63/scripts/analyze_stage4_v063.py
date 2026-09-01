#!/usr/bin/env python3
"""Validate v0.63 AHBN traces and aggregate Stage 4 result CSVs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd


METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")


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


def locate(root: Path, token: str) -> Path:
    matches = sorted((root / "outputs" / "csv").glob(f"*{token}*.csv"))
    if not matches:
        raise FileNotFoundError(f"no *{token}*.csv under {root / 'outputs/csv'}")
    return matches[-1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--experiment", required=True, choices=("exp07", "exp08", "exp09"))
    args = parser.parse_args()
    root = args.root.resolve()
    results_path = locate(root, f"{args.experiment}_results")
    trace_token = "exp08_ahbn_adaptive_trace" if args.experiment == "exp08" else f"{args.experiment}_adaptive_trace"
    trace_path = locate(root, trace_token)
    results = pd.read_csv(results_path)
    trace = pd.read_csv(trace_path)

    z = trace["score"].astype(float)
    expected = z.map(expected_fanout)
    actual = trace["fanout"].astype(int)
    weights = trace["weight"].astype(float)
    expected_mode = weights.map(lambda value: "gossip" if value >= 0.5 else "cluster")
    violations = int((actual != expected).sum())
    mode_violations = int((trace["mode"] != expected_mode).sum())
    if violations or mode_violations or not actual.isin(range(2, 7)).all():
        raise ValueError(
            f"trace validation failed: fanout={violations}, mode={mode_violations}, "
            f"invalid_gears={int((~actual.isin(range(2, 7))).sum())}"
        )

    condition = {
        "exp07": "fanout",
        "exp08": "ch_overload_factor",
        "exp09": "topology_param",
    }[args.experiment]
    grouped = []
    for keys, frame in results.groupby(["strategy", condition], dropna=False):
        row = {"strategy": keys[0], "condition": None if pd.isna(keys[1]) else keys[1], "n": len(frame)}
        for metric in METRICS:
            values = frame[metric].dropna().astype(float)
            n = len(values)
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            # Student-t 95% critical values for n<=30; asymptotic fallback thereafter.
            t95 = {1: math.nan, 2: 12.706, 3: 4.303, 4: 3.182, 5: 2.776,
                   6: 2.571, 7: 2.447, 8: 2.365, 9: 2.306, 10: 2.262,
                   20: 2.093}.get(n, 1.96 if n > 30 else 2.0)
            half = t95 * sd / math.sqrt(n) if n > 1 else math.nan
            mean = float(values.mean())
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd,
                        f"{metric}_ci95_low": mean - half, f"{metric}_ci95_high": mean + half})
        grouped.append(row)

    trace_summary = {
        "trace_rows": len(trace),
        "z_min": float(z.min()), "z_mean": float(z.mean()), "z_max": float(z.max()),
        **{f"{name}_min": float(trace[name].min()) for name in ("d_hat", "l_hat", "u_hat", "c_hat")},
        **{f"{name}_max": float(trace[name].max()) for name in ("d_hat", "l_hat", "u_hat", "c_hat")},
        **{f"fanout_{gear}_count": int((actual == gear).sum()) for gear in range(2, 7)},
        **{f"fanout_{gear}_proportion": float((actual == gear).mean()) for gear in range(2, 7)},
        "gossip_mode_count": int((trace["mode"] == "gossip").sum()),
        "gossip_mode_proportion": float((trace["mode"] == "gossip").mean()),
        "cluster_mode_count": int((trace["mode"] == "cluster").sum()),
        "cluster_mode_proportion": float((trace["mode"] == "cluster").mean()),
        "fanout_violations": violations, "mode_violations": mode_violations,
    }
    pd.DataFrame(grouped).to_csv(root / "aggregate_results.csv", index=False)
    (root / "ahbn_trace_summary.json").write_text(json.dumps(trace_summary, indent=2) + "\n")
    print(json.dumps({"validation": "PASS", "results": str(results_path),
                      "trace": str(trace_path), **trace_summary}, indent=2))


if __name__ == "__main__":
    main()
