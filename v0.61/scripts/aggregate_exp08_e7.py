#!/usr/bin/env python3
"""Validate and aggregate the four frozen Exp08 per-run datasets for E7."""

from __future__ import annotations

import hashlib
import math
from datetime import datetime
from pathlib import Path

import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
INPUTS = {
    "gossip": ROOT / "outputs/csv/exp08_gossip_results_20260820_111017.csv",
    "cluster": ROOT / "outputs/csv/exp08_structured_results_20260820_112714.csv",
    "dcsoc": ROOT / "outputs/csv/exp08_dcsoc_results_20260820_114555.csv",
    "ahbn": ROOT / "outputs/csv/exp08_ahbn_results_20260820_115817.csv",
}
METRICS = ["delivery_ratio", "propagation_delay", "duplicates", "total_forwards"]
SEEDS = set(range(42, 62))
OVERLOADS = {1.0, 1.5, 2.0, 3.0}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate(strategy: str, path: Path) -> pd.DataFrame:
    require(path.is_file(), f"{strategy}: input file does not exist: {path}")
    frame = pd.read_csv(path)
    require(len(frame) == 80, f"{strategy}: expected 80 rows, found {len(frame)}")
    require(set(frame["strategy"].unique()) == {strategy},
            f"{strategy}: unexpected strategy values {sorted(frame['strategy'].unique())}")
    require(set(frame["seed"]) == SEEDS and frame["seed"].nunique() == 20,
            f"{strategy}: seed set is not exactly 42..61")
    require(set(frame["ch_overload_factor"].astype(float)) == OVERLOADS,
            f"{strategy}: overload set is not exactly {sorted(OVERLOADS)}")
    counts = frame.groupby("ch_overload_factor").size()
    require(len(counts) == 4 and (counts == 20).all(),
            f"{strategy}: expected 20 observations per overload, found {counts.to_dict()}")
    require(not frame.duplicated(["seed", "ch_overload_factor"]).any(),
            f"{strategy}: duplicate (seed, ch_overload_factor) combinations found")
    missing = [metric for metric in METRICS if metric not in frame.columns]
    require(not missing, f"{strategy}: missing required metrics {missing}")
    for metric in METRICS:
        values = pd.to_numeric(frame[metric], errors="coerce")
        require(values.notna().all() and values.map(math.isfinite).all(),
                f"{strategy}: metric {metric} contains non-numeric or non-finite values")
        frame[metric] = values
    expected_grid = {(seed, overload) for seed in SEEDS for overload in OVERLOADS}
    actual_grid = set(zip(frame["seed"], frame["ch_overload_factor"].astype(float)))
    require(actual_grid == expected_grid, f"{strategy}: incomplete experimental grid")
    return frame


def main() -> None:
    print("E7 Exp08 aggregation")
    print("Input datasets:")
    for strategy, path in INPUTS.items():
        print(f"  {strategy}: {path.relative_to(ROOT)}")

    before = {path: sha256(path) for path in INPUTS.values()}
    frames = [validate(strategy, path) for strategy, path in INPUTS.items()]
    raw = pd.concat(frames, ignore_index=True)
    require(len(raw) == 320, f"cross-dataset: expected 320 rows, found {len(raw)}")

    grids = [set(zip(frame["seed"], frame["ch_overload_factor"].astype(float))) for frame in frames]
    require(all(grid == grids[0] for grid in grids[1:]),
            "cross-dataset: strategies do not share the same complete experimental grid")

    rows = []
    for (strategy, overload), group in raw.groupby(["strategy", "ch_overload_factor"], sort=True):
        row = {"strategy": strategy, "ch_overload_factor": overload, "n": len(group)}
        for metric in METRICS:
            values = group[metric]
            n = len(values)
            mean = values.mean()
            std = values.std(ddof=1)
            sem = std / math.sqrt(n)
            margin = t.ppf(0.975, df=n - 1) * sem
            row.update({
                f"{metric}_mean": mean,
                f"{metric}_std": std,
                f"{metric}_sem": sem,
                f"{metric}_ci95_low": mean - margin,
                f"{metric}_ci95_high": mean + margin,
            })
        rows.append(row)

    summary = pd.DataFrame(rows)
    require(len(summary) == 16, f"expected 16 aggregate conditions, found {len(summary)}")
    require((summary["n"] == 20).all(), "one or more aggregate rows do not have n=20")
    expected_conditions = {(s, o) for s in INPUTS for o in OVERLOADS}
    actual_conditions = set(zip(summary["strategy"], summary["ch_overload_factor"].astype(float)))
    require(actual_conditions == expected_conditions, "aggregate condition set is incomplete or contains extras")
    for metric in METRICS:
        mean = summary[f"{metric}_mean"]
        std = summary[f"{metric}_std"]
        sem = summary[f"{metric}_sem"]
        low = summary[f"{metric}_ci95_low"]
        high = summary[f"{metric}_ci95_high"]
        require(all(series.map(math.isfinite).all() for series in (mean, std, sem, low, high)),
                f"{metric}: aggregate statistics contain non-finite values")
        require((std >= 0).all() and (sem >= 0).all(), f"{metric}: negative std or SEM")
        require(((low <= mean) & (mean <= high)).all(), f"{metric}: CI does not contain mean")

    output = ROOT / "outputs/csv" / f"exp08_summary_{datetime.now():%Y%m%d_%H%M%S}.csv"
    summary.to_csv(output, index=False)
    after = {path: sha256(path) for path in INPUTS.values()}
    require(before == after, "one or more frozen source CSVs changed during aggregation")

    print("\nCompact aggregation table (mean [95% CI]):")
    for _, row in summary.iterrows():
        stats = "  ".join(
            f"{metric}={row[f'{metric}_mean']:.6g} "
            f"[{row[f'{metric}_ci95_low']:.6g}, {row[f'{metric}_ci95_high']:.6g}]"
            for metric in METRICS
        )
        print(f"{row['strategy']:<8} overload={row['ch_overload_factor']:<3g} n={int(row['n'])}  {stats}")

    print("\nStrategies: 4")
    print("Seeds per condition: 20")
    print(f"Overload factors: {sorted(OVERLOADS)}")
    print("Expected raw rows: 320")
    print(f"Validated raw rows: {len(raw)}")
    print("Expected aggregate conditions: 16")
    print(f"Generated aggregate conditions: {len(summary)}")
    print("95% CI method: Student t")
    print("Degrees of freedom per condition: 19")
    print(f"Metrics aggregated: {METRICS}")
    print(f"Saved: {output.relative_to(ROOT)}")
    print("Overall E7: PASS")


if __name__ == "__main__":
    main()
