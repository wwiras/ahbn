#!/usr/bin/env python3
"""Generate Exp09 v0.63 figures from validated machine-readable summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


STRATEGIES = {"gossip": "Gossip", "cluster": "Structured", "dcsoc": "DC-SoC", "ahbn": "AHBN"}
P_VALUES = [0.04, 0.06, 0.08, 0.10, 0.12]
STYLES = {"gossip": ("o", "-"), "cluster": ("s", "--"), "dcsoc": ("^", "-."), "ahbn": ("D", ":")}
METRICS = (
    ("delivery_ratio", "Delivery Ratio", "exp09_v063_delivery.png"),
    ("propagation_delay", "Propagation Delay (s)", "exp09_v063_delay.png"),
    ("duplicates", "Duplicate Transmissions", "exp09_v063_duplicates.png"),
    ("total_forwards", "Total Forwarding Transmissions", "exp09_v063_forwards.png"),
)


def require(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)


def metric_plot(root: Path, frame: pd.DataFrame, metric: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for strategy, label in STRATEGIES.items():
        rows = frame[frame["strategy"] == strategy].sort_values("density_p")
        x, mean = rows["density_p"].astype(float), rows[f"{metric}_mean"].astype(float)
        low, high = rows[f"{metric}_ci95_low"].astype(float), rows[f"{metric}_ci95_high"].astype(float)
        marker, linestyle = STYLES[strategy]
        ax.errorbar(x, mean, yerr=[mean-low, high-mean], marker=marker, linestyle=linestyle,
                    linewidth=1.9, markersize=6.5, capsize=4, label=label)
    ax.set_xlabel("ER Edge Probability (p)"); ax.set_ylabel(ylabel); ax.set_title(f"Exp09 v0.63 — {ylabel}")
    ax.set_xticks(P_VALUES); ax.grid(True, linestyle=":", alpha=0.7); ax.legend(ncols=2)
    fig.tight_layout(); output = root / filename; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def z_plot(root: Path, adaptive: pd.DataFrame) -> Path:
    x, mean = adaptive["density_p"].astype(float), adaptive["z_mean"].astype(float)
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(x, mean, "D-", color="tab:orange", linewidth=2, label="Mean z")
    ax.fill_between(x, adaptive["z_min"].astype(float), adaptive["z_max"].astype(float),
                    color="tab:orange", alpha=0.18, label="Observed min–max")
    ax.axhline(-0.25, color="gray", linestyle="--", linewidth=1.2, label="Fanout-2 boundary")
    ax.axhline(0.25, color="gray", linestyle=":", linewidth=1.2, label="Fanout-4 boundary")
    ax.set_xlabel("ER Edge Probability (p)"); ax.set_ylabel("Controller Score z")
    ax.set_title("Exp09 v0.63 — AHBN Controller Score by Density"); ax.set_xticks(P_VALUES)
    ax.grid(True, linestyle=":", alpha=0.7); ax.legend(); fig.tight_layout()
    output = root / "exp09_v063_ahbn_z_by_density.png"; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def stacked(root: Path, adaptive: pd.DataFrame, kind: str) -> Path:
    x = adaptive["density_p"].astype(float); bottom = [0.0] * len(adaptive)
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    if kind == "fanout":
        series = [(f"Fanout {k}", adaptive[f"fanout_{k}_proportion"].astype(float)*100) for k in range(2, 7)]
        filename, title = "exp09_v063_ahbn_fanout_by_density.png", "AHBN Fanout Distribution by Density"
    else:
        series = [("Gossip mode", adaptive["gossip_mode_proportion"].astype(float)*100),
                  ("Cluster mode", adaptive["cluster_mode_proportion"].astype(float)*100)]
        filename, title = "exp09_v063_ahbn_mode_by_density.png", "AHBN Mode Distribution by Density"
    for label, values in series:
        ax.bar(x, values, width=0.013, bottom=bottom, label=label, edgecolor="white", linewidth=0.5)
        bottom = [a+b for a, b in zip(bottom, values)]
    ax.set_xlabel("ER Edge Probability (p)"); ax.set_ylabel("Controller Decisions (%)")
    ax.set_title(f"Exp09 v0.63 — {title}"); ax.set_xticks(P_VALUES); ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.7); ax.legend(ncols=3 if kind == "fanout" else 2, fontsize=9)
    fig.tight_layout(); output = root / filename; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def topology_plot(root: Path, topology: pd.DataFrame) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(topology["density_p"], topology["mean_degree"], "o-", linewidth=2)
    ax.set_xlabel("Configured ER Edge Probability (p)"); ax.set_ylabel("Realized Mean Degree")
    ax.set_title("Exp09 v0.63 — Realized ER Density"); ax.set_xticks(P_VALUES); ax.grid(True, linestyle=":", alpha=0.7)
    fig.tight_layout(); output = root / "exp09_v063_realized_mean_degree.png"; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("formal_output", type=Path); args = parser.parse_args()
    root = args.formal_output.resolve(); require(root.is_dir(), f"missing formal output: {root}")
    summary = pd.read_csv(root / "exp09_v063_summary.csv")
    adaptive = pd.read_csv(root / "exp09_v063_ahbn_adaptive_summary.csv").sort_values("density_p")
    topology = pd.read_csv(root / "exp09_v063_topology_summary.csv").sort_values("density_p")
    require(len(summary) == 20 and set(summary["strategy"]) == set(STRATEGIES), "summary grid mismatch")
    summary_density = sorted(pd.to_numeric(summary["density_p"], errors="coerce").unique())
    summary_n = pd.to_numeric(summary["n"], errors="coerce")
    require(len(summary_density) == len(P_VALUES) and np.allclose(summary_density, P_VALUES)
            and summary_n.eq(20).all(), "density/n mismatch")
    adaptive_density = sorted(pd.to_numeric(adaptive["density_p"], errors="coerce").unique())
    require(len(adaptive) == 5 and len(adaptive_density) == len(P_VALUES)
            and np.allclose(adaptive_density, P_VALUES), "adaptive grid mismatch")
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10})
    outputs = [metric_plot(root, summary, *item) for item in METRICS]
    outputs += [z_plot(root, adaptive), stacked(root, adaptive, "fanout"), stacked(root, adaptive, "mode"), topology_plot(root, topology)]
    for output in outputs:
        require(output.is_file() and output.stat().st_size > 0, f"missing/empty figure: {output}")
        print(f"Generated figure: {output}")


if __name__ == "__main__": main()
