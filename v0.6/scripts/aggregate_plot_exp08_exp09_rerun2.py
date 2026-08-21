"""Aggregate and plot only validated Stage 4 rerun-2 Exp08/Exp09 CSVs."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
METRICS = {
    "delivery_ratio": "Delivery ratio",
    "propagation_delay": "Propagation delay (s)",
    "duplicates": "Duplicates",
    "total_forwards": "Total forwards",
}
NAMES = {"gossip": "Gossip", "cluster": "Structured", "dcsoc": "DC-SoC", "ahbn": "AHBN"}
STYLES = [("o", "-"), ("s", "--"), ("^", "-."), ("D", ":")]


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def aggregate(
    experiment: str,
    source: Path,
    condition: str,
    levels: list[float],
    seeds: set[int],
    timestamp: str,
) -> Path:
    before = digest(source)
    frame = pd.read_csv(source)
    expected = {(strategy, level, seed) for strategy in NAMES for level in levels for seed in seeds}
    keys = ["strategy", condition, "seed"]
    require(len(frame) == len(expected), f"{experiment}: wrong row count")
    require(set(map(tuple, frame[keys].itertuples(index=False, name=None))) == expected, f"{experiment}: incomplete grid")
    require(not frame.duplicated(keys).any(), f"{experiment}: duplicate identity")
    for metric in METRICS:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        require(frame[metric].notna().all() and frame[metric].map(math.isfinite).all(), f"{experiment}: invalid {metric}")

    rows = []
    for (strategy, level), group in frame.groupby(["strategy", condition], sort=True):
        row = {"comparator": NAMES[strategy], "strategy": strategy, condition: level, "n": len(group), "df": len(group) - 1}
        for metric in METRICS:
            mean = group[metric].mean()
            sd = group[metric].std(ddof=1)
            se = sd / math.sqrt(len(group))
            ci = t.ppf(0.975, len(group) - 1) * se
            row.update({f"{metric}_mean": mean, f"{metric}_sd": sd, f"{metric}_se": se, f"{metric}_ci95": ci})
        rows.append(row)
    summary = pd.DataFrame(rows)
    require(len(summary) == len(NAMES) * len(levels), f"{experiment}: bad summary grid")
    destination = ROOT / "outputs" / "csv" / f"{experiment}_rerun2_summary_{timestamp}.csv"
    summary.to_csv(destination, index=False)
    require(digest(source) == before, f"{experiment}: source changed during aggregation")
    print(f"{experiment.upper()} input: {source}")
    print(f"{experiment.upper()} input SHA-256: {before}")
    print(f"{experiment.upper()} summary: {destination}")
    return destination


def plot(experiment: str, summary_path: Path, condition: str, levels: list[float], xlabel: str, timestamp: str) -> list[Path]:
    frame = pd.read_csv(summary_path)
    outputs = []
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]
    for metric, ylabel in METRICS.items():
        figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
        for comparator, color, (marker, linestyle) in zip(NAMES.values(), colors, STYLES):
            part = frame[frame["comparator"] == comparator].sort_values(condition)
            require(len(part) == len(levels), f"{experiment}: bad plot grid for {comparator}")
            axis.errorbar(part[condition], part[f"{metric}_mean"], yerr=part[f"{metric}_ci95"], label=comparator, color=color, marker=marker, linestyle=linestyle, capsize=3)
        axis.set_xlabel(xlabel)
        axis.set_ylabel(ylabel)
        axis.set_xticks(levels)
        axis.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)
        axis.legend(frameon=False, ncols=2)
        destination = ROOT / "outputs" / "figures" / f"{experiment}_rerun2_{metric}_{timestamp}.png"
        destination.parent.mkdir(parents=True, exist_ok=True)
        figure.savefig(destination, dpi=300, bbox_inches="tight")
        plt.close(figure)
        require(destination.is_file() and destination.stat().st_size > 0, f"missing plot {destination}")
        outputs.append(destination)
        print(f"{experiment.upper()} plot: {destination}")
    return outputs


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp08", type=Path, required=True)
    parser.add_argument("--exp09", type=Path, required=True)
    parser.add_argument("--timestamp", required=True)
    args = parser.parse_args()
    exp08 = args.exp08.resolve()
    exp09 = args.exp09.resolve()
    exp08_summary = aggregate("exp08", exp08, "ch_overload_factor", [1.0, 1.5, 2.0, 3.0], set(range(42, 62)), args.timestamp)
    exp09_summary = aggregate("exp09", exp09, "topology_param", [0.04, 0.06, 0.08, 0.10, 0.12], set(range(42, 72)), args.timestamp)
    plot("exp08", exp08_summary, "ch_overload_factor", [1.0, 1.5, 2.0, 3.0], "CH overload factor", args.timestamp)
    plot("exp09", exp09_summary, "topology_param", [0.04, 0.06, 0.08, 0.10, 0.12], "ER edge probability", args.timestamp)
    print("EXP08/EXP09 RERUN-2 AGGREGATION AND PLOTTING: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
