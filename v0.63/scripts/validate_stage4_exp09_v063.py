#!/usr/bin/env python3
"""Validate an Exp09 v0.63 smoke or formal dataset before analysis."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd
import yaml


P_VALUES = [0.04, 0.06, 0.08, 0.10, 0.12]
STRATEGIES = ["gossip", "cluster", "dcsoc", "ahbn"]
METRICS = ["delivery_ratio", "propagation_delay", "duplicates", "total_forwards"]


def fanout_for(z: float) -> int:
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
    if len(matches) != 1:
        raise ValueError(f"expected exactly one *{token}*.csv, found {len(matches)}")
    return matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    expected_design = {
        "experiment": "exp09", "num_nodes": 100, "topology_type": "er",
        "edge_probs": P_VALUES, "num_clusters": 4, "base_delay": 1.0,
        "jitter": 0.2, "message_source": 0, "strategies": STRATEGIES,
    }
    for key, expected in expected_design.items():
        if cfg.get(key) != expected:
            raise ValueError(f"frozen design mismatch for {key}: {cfg.get(key)!r} != {expected!r}")
    ahbn = cfg["ahbn"]
    expected_ahbn = {
        "alpha": 0.3, "d0": 0.0, "l0": 0.0, "u0": 0.0, "c0": 0.0,
        "w_d": -1.0, "w_l": 1.0, "w_u": 1.0, "w_c": 1.0,
        "min_fanout": 2, "max_fanout": 6, "mode_threshold": 0.5,
        "default_fanout": 3,
    }
    for key, expected in expected_ahbn.items():
        if ahbn.get(key) != expected:
            raise ValueError(f"canonical AHBN mismatch for {key}: {ahbn.get(key)!r} != {expected!r}")

    results = pd.read_csv(locate(root, "exp09_results"))
    trace = pd.read_csv(locate(root, "exp09_adaptive_trace"))
    required = {"experiment", "strategy", "seed", "num_nodes", "topology_type",
                "topology_param", *METRICS}
    if not required.issubset(results.columns):
        raise ValueError(f"result schema missing: {sorted(required - set(results.columns))}")
    if set(results["strategy"]) != set(STRATEGIES):
        raise ValueError(f"strategy set mismatch: {sorted(set(results['strategy']))}")
    if set(results["topology_type"]) != {"er"}:
        raise ValueError("Exp09 topology is not exclusively ER")
    if sorted(results["topology_param"].astype(float).unique()) != P_VALUES:
        raise ValueError("Exp09 density levels mismatch")
    runs = int(cfg["runs_per_setting"])
    counts = results.groupby(["strategy", "topology_param"]).size()
    if len(counts) != 20 or not (counts == runs).all():
        raise ValueError(f"incomplete Exp09 grid: {counts.to_dict()}")
    if results.duplicated(["strategy", "topology_param", "seed"]).any():
        raise ValueError("duplicate strategy/density/seed combination")
    numeric = results[METRICS].apply(pd.to_numeric, errors="coerce")
    if numeric.isna().any().any() or not numeric.map(math.isfinite).all().all():
        raise ValueError("NaN or nonfinite primary metric")
    if not numeric["delivery_ratio"].between(0, 1).all():
        raise ValueError("delivery outside [0,1]")
    if not (numeric[["propagation_delay", "duplicates", "total_forwards"]] >= 0).all().all():
        raise ValueError("negative primary metric")

    trace_required = {"scenario_tag", "score", "weight", "mode", "fanout",
                      "d_hat", "l_hat", "u_hat", "c_hat"}
    if not trace_required.issubset(trace.columns):
        raise ValueError(f"trace schema missing: {sorted(trace_required - set(trace.columns))}")
    trace_numeric = trace[["score", "weight", "fanout", "d_hat", "l_hat", "u_hat", "c_hat"]].apply(
        pd.to_numeric, errors="coerce")
    if trace_numeric.isna().any().any() or not trace_numeric.map(math.isfinite).all().all():
        raise ValueError("NaN or nonfinite AHBN trace value")
    expected_fanout = trace_numeric["score"].map(fanout_for)
    fanout_violations = int((trace_numeric["fanout"].astype(int) != expected_fanout).sum())
    expected_mode = trace_numeric["weight"].map(lambda value: "gossip" if value >= 0.5 else "cluster")
    mode_violations = int((trace["mode"] != expected_mode).sum())
    score_error = (trace_numeric["score"] - (
        -trace_numeric["d_hat"] + trace_numeric["l_hat"]
        + trace_numeric["u_hat"] + trace_numeric["c_hat"])).abs()
    score_violations = int((score_error > 1e-12).sum())
    if fanout_violations or mode_violations or score_violations:
        raise ValueError(f"controller violations: score={score_violations}, "
                         f"fanout={fanout_violations}, mode={mode_violations}")

    topology_rows = []
    for p in P_VALUES:
        for seed in sorted(results.loc[results["topology_param"].astype(float) == p, "seed"].unique()):
            path = root / "outputs" / "topologies" / f"er_n100_p{p}_seed{int(seed)}.json"
            if not path.is_file():
                raise FileNotFoundError(f"missing topology evidence: {path}")
            payload = json.loads(path.read_text(encoding="utf-8"))
            node_count = len(payload["nodes"])
            edge_count = len(payload["edges"])
            topology_rows.append({"density_p": p, "seed": int(seed), "node_count": node_count,
                                  "edge_count": edge_count,
                                  "realized_mean_degree": 2 * edge_count / node_count})
    topology = pd.DataFrame(topology_rows)
    diagnostics = topology.groupby("density_p", as_index=False).agg(
        runs=("seed", "size"), mean_nodes=("node_count", "mean"),
        mean_edge_count=("edge_count", "mean"),
        mean_degree=("realized_mean_degree", "mean"))
    if not diagnostics["mean_edge_count"].is_monotonic_increasing:
        raise ValueError("realized mean edge count does not increase with configured p")
    topology.to_csv(root / "exp09_v063_topology_evidence.csv", index=False)
    diagnostics.to_csv(root / "exp09_v063_topology_summary.csv", index=False)

    print("EXP09 DATASET / TOPOLOGY AUDIT: PASS")
    print(f"expected runs: {len(P_VALUES) * len(STRATEGIES) * runs}")
    print(f"actual runs: {len(results)}")
    print(f"cells: {len(counts)}; replicates per cell: {runs}")
    print(f"AHBN trace rows: {len(trace)}")
    print(f"controller violations: score={score_violations}, fanout={fanout_violations}, mode={mode_violations}")
    print("realized topology by configured p:")
    print(diagnostics.to_string(index=False))


if __name__ == "__main__":
    main()
