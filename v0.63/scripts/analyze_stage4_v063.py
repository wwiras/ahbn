#!/usr/bin/env python3
"""Validate v0.63 AHBN traces and aggregate Stage 4 result CSVs."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
from scipy.stats import t


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
    condition = {
        "exp07": "fanout",
        "exp08": "ch_overload_factor",
        "exp09": "topology_param",
    }[args.experiment]

    required_metrics = list(METRICS)
    required_results = {"experiment", "strategy", "seed", condition, *required_metrics}
    if not required_results.issubset(results.columns):
        raise ValueError(f"results missing columns: {sorted(required_results - set(results.columns))}")
    numeric_metrics = results[required_metrics].apply(pd.to_numeric, errors="coerce")
    if numeric_metrics.isna().any().any() or not numeric_metrics.map(math.isfinite).all().all():
        raise ValueError("results contain missing/nonfinite primary metrics")
    if not numeric_metrics["delivery_ratio"].between(0.0, 1.0).all():
        raise ValueError("delivery_ratio outside [0,1]")
    if not (numeric_metrics[["propagation_delay", "duplicates", "total_forwards"]] >= 0).all().all():
        raise ValueError("negative primary metric")

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

    if args.experiment == "exp08":
        strategies = {"gossip", "cluster", "dcsoc", "ahbn"}
        overloads = {1.0, 1.5, 2.0, 3.0}
        if set(results["strategy"]) != strategies or set(results[condition].astype(float)) != overloads:
            raise ValueError("Exp08 strategy/overload treatment set mismatch")
        counts = results.groupby(["strategy", condition]).size()
        if len(counts) != 16 or counts.nunique() != 1 or int(counts.iloc[0]) not in {1, 20}:
            raise ValueError(f"Exp08 incomplete/inconsistent run grid: {counts.to_dict()}")
        if results.duplicated(["strategy", condition, "seed"]).any():
            raise ValueError("Exp08 duplicate strategy/overload/seed run")
        evidence_path = locate(root, "exp08_execution_evidence")
        evidence = pd.read_csv(evidence_path)
        evidence_keys = ["strategy", "overload_factor", "seed"]
        if len(evidence) != len(results) or evidence.duplicated(evidence_keys).any():
            raise ValueError("Exp08 evidence row/key mismatch")
        result_keys = set(zip(results["strategy"], results[condition].astype(float), results["seed"].astype(int)))
        evidence_key_set = set(zip(evidence["strategy"], evidence["overload_factor"].astype(float), evidence["seed"].astype(int)))
        if result_keys != evidence_key_set:
            raise ValueError("Exp08 result/evidence keys do not join one-to-one")
        topology_counts = evidence.groupby(["overload_factor", "seed"])["topology_identity"].nunique()
        if not (topology_counts == 1).all():
            raise ValueError("Exp08 topology identity mismatch across strategies")
        dcsoc = evidence[evidence["strategy"] == "dcsoc"]
        for index, row in dcsoc.iterrows():
            eligible = json.loads(row["dcsoc_eligible_overload_nodes"])
            target = int(row["dcsoc_selected_overload_node"])
            if (int(row["effective_message_source"]) != int(row["dcsoc_master"])
                    or target not in eligible
                    or row["dcsoc_selected_overload_role"] not in {"Master", "Core"}
                    or int(row["max_structural_obligations"]) <= 3):
                raise ValueError(f"Exp08 DC-SoC overload/source contract violation at evidence row {index + 2}")
        print(f"Exp08 evidence: PASS ({len(evidence)} rows; paired topologies; DC-SoC Master/Core targets)")
    grouped = []
    for keys, frame in results.groupby(["strategy", condition], dropna=False):
        row = {"strategy": keys[0], "condition": None if pd.isna(keys[1]) else keys[1], "n": len(frame)}
        for metric in METRICS:
            values = frame[metric].dropna().astype(float)
            n = len(values)
            sd = float(values.std(ddof=1)) if n > 1 else math.nan
            t95 = float(t.ppf(0.975, n - 1)) if n > 1 else math.nan
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
