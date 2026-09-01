#!/usr/bin/env python3
"""Generate publication-ready figures from frozen Exp07 v0.63 summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


EXPECTED = [*(f"Gossip f{k}" for k in range(2, 7)), "AHBN adaptive"]
METRICS = (
    ("delivery", "Delivery Ratio", "exp07_v063_delivery.png"),
    ("delay", "Propagation Delay (s)", "exp07_v063_delay.png"),
    ("duplicates", "Duplicate Transmissions", "exp07_v063_duplicates.png"),
    ("forwards", "Total Forwarding Transmissions", "exp07_v063_forwards.png"),
)


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def load_inputs(root: Path) -> tuple[pd.DataFrame, pd.Series]:
    summary_path = root / "exp07_v063_summary.csv"
    adaptive_path = root / "exp07_v063_ahbn_adaptive_summary.csv"
    require(summary_path.is_file(), f"missing summary: {summary_path}")
    require(adaptive_path.is_file(), f"missing adaptive summary: {adaptive_path}")
    summary = pd.read_csv(summary_path)
    adaptive = pd.read_csv(adaptive_path)
    required = {"treatment", "strategy", "fanout", "n"}
    for short, _, _ in METRICS:
        required.update({f"{short}_mean", f"{short}_ci95_low", f"{short}_ci95_high"})
    require(required <= set(summary.columns), f"summary missing columns: {sorted(required - set(summary.columns))}")
    require(summary["treatment"].tolist() == EXPECTED, f"unexpected treatment order/set: {summary['treatment'].tolist()}")
    require((summary["n"] == 20).all(), "all treatments must have n=20")
    require(len(adaptive) == 1, "adaptive summary must contain exactly one row")
    adaptive_required = {*(f"fanout_{k}_proportion" for k in range(2, 7)),
                         "gossip_mode_proportion", "cluster_mode_proportion"}
    require(adaptive_required <= set(adaptive.columns),
            f"adaptive summary missing columns: {sorted(adaptive_required - set(adaptive.columns))}")
    return summary, adaptive.iloc[0]


def style() -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 11, "axes.labelsize": 12,
        "axes.titlesize": 12, "legend.fontsize": 10, "xtick.labelsize": 10,
        "ytick.labelsize": 10, "axes.linewidth": 1.0, "lines.linewidth": 2.0,
        "lines.markersize": 7, "figure.dpi": 120,
    })


def metric_figure(root: Path, summary: pd.DataFrame, short: str, ylabel: str, filename: str) -> Path:
    gossip = summary[summary["strategy"] == "gossip"].sort_values("fanout")
    ahbn = summary[summary["strategy"] == "ahbn"].iloc[0]
    require(gossip["fanout"].astype(int).tolist() == [2, 3, 4, 5, 6], "Gossip fanout cells incomplete")
    x = gossip["fanout"].astype(int)
    mean = gossip[f"{short}_mean"].astype(float)
    lower = mean - gossip[f"{short}_ci95_low"].astype(float)
    upper = gossip[f"{short}_ci95_high"].astype(float) - mean
    ahbn_mean = float(ahbn[f"{short}_mean"])
    ahbn_low = float(ahbn[f"{short}_ci95_low"])
    ahbn_high = float(ahbn[f"{short}_ci95_high"])

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    ax.errorbar(x, mean, yerr=[lower, upper], fmt="o-", color="tab:blue",
                capsize=5, capthick=1.2, label="Gossip (fixed fanout)", zorder=3)
    ax.axhspan(ahbn_low, ahbn_high, color="tab:orange", alpha=0.18,
               label="AHBN adaptive 95% CI")
    ax.axhline(ahbn_mean, color="tab:orange", linestyle="--", linewidth=2.0,
               label="AHBN adaptive mean")
    ax.set_xlabel("Fixed Gossip Fanout")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Exp07 v0.63 — {ylabel}")
    ax.set_xticks([2, 3, 4, 5, 6])
    ax.grid(True, linestyle=":", linewidth=0.8, alpha=0.7)
    ax.legend(frameon=True)
    fig.tight_layout()
    output = root / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def distribution_figure(root: Path, adaptive: pd.Series, kind: str) -> Path:
    if kind == "fanout":
        labels = [f"Fanout {k}" for k in range(2, 7)]
        values = [100 * float(adaptive[f"fanout_{k}_proportion"]) for k in range(2, 7)]
        filename, title, color = "exp07_v063_ahbn_fanout_distribution.png", "AHBN Realized Fanout Distribution", "tab:purple"
    else:
        labels = ["Gossip mode", "Cluster mode"]
        values = [100 * float(adaptive["gossip_mode_proportion"]),
                  100 * float(adaptive["cluster_mode_proportion"])]
        filename, title, color = "exp07_v063_ahbn_mode_distribution.png", "AHBN Realized Mode Distribution", "tab:green"
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    bars = ax.bar(labels, values, color=color, alpha=0.82, edgecolor="black", linewidth=0.7)
    ax.set_ylabel("Controller Decisions (%)")
    ax.set_title(f"Exp07 v0.63 — {title}")
    ax.set_ylim(0, max(100, max(values) * 1.13))
    ax.grid(axis="y", linestyle=":", linewidth=0.8, alpha=0.7)
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
                f"{value:.2f}%", ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    output = root / filename
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("formal_output", type=Path)
    args = parser.parse_args()
    root = args.formal_output.resolve()
    require(root.is_dir(), f"formal output directory does not exist: {root}")
    require((root / "terminal.log").is_file(), f"missing formal terminal log: {root / 'terminal.log'}")
    require(" formal" in (root / "terminal.log").read_text(encoding="utf-8").splitlines()[0],
            f"not a formal Exp07 output: {root}")
    summary, adaptive = load_inputs(root)
    style()
    outputs = [metric_figure(root, summary, *metric) for metric in METRICS]
    outputs.append(distribution_figure(root, adaptive, "fanout"))
    outputs.append(distribution_figure(root, adaptive, "mode"))
    for output in outputs:
        require(output.is_file() and output.stat().st_size > 0, f"empty/missing generated figure: {output}")
        print(f"Generated figure: {output}")


if __name__ == "__main__":
    main()
