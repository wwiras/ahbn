#!/usr/bin/env python3
"""Generate Exp08 v0.63 figures from validated machine-readable summaries."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


STRATEGIES = {"gossip": "Gossip", "cluster": "Structured", "dcsoc": "DC-SoC", "ahbn": "AHBN"}
OVERLOADS = [1.0, 1.5, 2.0, 3.0]
METRICS = (
    ("propagation_delay", "Propagation Delay (s)", "exp08_v063_delay.png"),
    ("delivery_ratio", "Delivery Ratio", "exp08_v063_delivery.png"),
    ("duplicates", "Duplicate Transmissions", "exp08_v063_duplicates.png"),
    ("total_forwards", "Total Forwarding Transmissions", "exp08_v063_forwards.png"),
)
STYLES = {"gossip": ("o", "-"), "cluster": ("s", "--"), "dcsoc": ("^", "-."), "ahbn": ("D", ":")}


def require(ok: bool, message: str) -> None:
    if not ok: raise ValueError(message)


def load(root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    summary_path = root / "exp08_v063_summary.csv"
    adaptive_path = root / "exp08_v063_ahbn_adaptive_summary.csv"
    require(summary_path.is_file() and adaptive_path.is_file(), "missing Exp08 analysis summaries")
    summary, adaptive = pd.read_csv(summary_path), pd.read_csv(adaptive_path)
    required = {"strategy", "ch_overload_factor", "n"}
    for metric, _, _ in METRICS:
        required.update({f"{metric}_mean", f"{metric}_ci95_low", f"{metric}_ci95_high"})
    require(required <= set(summary.columns), f"summary columns missing: {sorted(required - set(summary.columns))}")
    require(len(summary) == 16 and set(summary.strategy) == set(STRATEGIES), "summary treatment grid mismatch")
    require(set(summary.ch_overload_factor.astype(float)) == set(OVERLOADS) and (summary.n == 20).all(), "summary overload/n mismatch")
    require(len(adaptive) == 4 and set(adaptive.ch_overload_factor.astype(float)) == set(OVERLOADS), "adaptive overload grid mismatch")
    return summary, adaptive.sort_values("ch_overload_factor")


def metric_plot(root: Path, frame: pd.DataFrame, metric: str, ylabel: str, filename: str) -> Path:
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    for strategy, label in STRATEGIES.items():
        rows = frame[frame.strategy == strategy].sort_values("ch_overload_factor")
        x, mean = rows.ch_overload_factor.astype(float), rows[f"{metric}_mean"].astype(float)
        low, high = rows[f"{metric}_ci95_low"].astype(float), rows[f"{metric}_ci95_high"].astype(float)
        marker, linestyle = STYLES[strategy]
        ax.errorbar(x, mean, yerr=[mean - low, high - mean], marker=marker, linestyle=linestyle,
                    linewidth=1.9, markersize=6.5, capsize=4, label=label)
    ax.set_xlabel("CH Overload Factor"); ax.set_ylabel(ylabel); ax.set_title(f"Exp08 v0.63 — {ylabel}")
    ax.set_xticks(OVERLOADS); ax.grid(True, linestyle=":", alpha=0.7); ax.legend(ncols=2)
    fig.tight_layout(); output = root / filename; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def z_plot(root: Path, adaptive: pd.DataFrame) -> Path:
    x = adaptive.ch_overload_factor.astype(float); mean = adaptive.z_mean.astype(float)
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    ax.plot(x, mean, "D-", color="tab:orange", linewidth=2, label="Mean z")
    ax.fill_between(x, adaptive.z_min.astype(float), adaptive.z_max.astype(float), color="tab:orange", alpha=0.18, label="Observed min–max")
    ax.axhline(0.25, color="gray", linestyle="--", linewidth=1.3, label="Fanout-4 threshold")
    ax.set_xlabel("CH Overload Factor"); ax.set_ylabel("Controller Score z")
    ax.set_title("Exp08 v0.63 — AHBN Controller Score by Overload"); ax.set_xticks(OVERLOADS)
    ax.grid(True, linestyle=":", alpha=0.7); ax.legend(); fig.tight_layout()
    output = root / "exp08_v063_ahbn_z_by_overload.png"; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def stacked(root: Path, adaptive: pd.DataFrame, kind: str) -> Path:
    x = adaptive.ch_overload_factor.astype(float); bottom = [0.0] * len(adaptive)
    fig, ax = plt.subplots(figsize=(7.4, 4.9))
    if kind == "fanout":
        series = [(f"Fanout {k}", adaptive[f"fanout_{k}_proportion"].astype(float) * 100) for k in range(2, 7)]
        filename, title = "exp08_v063_ahbn_fanout_by_overload.png", "AHBN Fanout Distribution by Overload"
    else:
        series = [("Gossip mode", adaptive.gossip_mode_proportion.astype(float) * 100),
                  ("Cluster mode", adaptive.cluster_mode_proportion.astype(float) * 100)]
        filename, title = "exp08_v063_ahbn_mode_by_overload.png", "AHBN Mode Distribution by Overload"
    for label, values in series:
        ax.bar(x, values, width=0.26, bottom=bottom, label=label, edgecolor="white", linewidth=0.5)
        bottom = [a + b for a, b in zip(bottom, values)]
    ax.set_xlabel("CH Overload Factor"); ax.set_ylabel("Controller Decisions (%)")
    ax.set_title(f"Exp08 v0.63 — {title}"); ax.set_xticks(OVERLOADS); ax.set_ylim(0, 100)
    ax.grid(axis="y", linestyle=":", alpha=0.7); ax.legend(ncols=3 if kind == "fanout" else 2, fontsize=9)
    fig.tight_layout(); output = root / filename; fig.savefig(output, dpi=300, bbox_inches="tight"); plt.close(fig); return output


def main() -> None:
    parser = argparse.ArgumentParser(); parser.add_argument("formal_output", type=Path); args = parser.parse_args()
    root = args.formal_output.resolve(); require(root.is_dir(), f"missing formal output: {root}")
    log = root / "terminal.log"; require(log.is_file() and "Stage 4 exp08 ControlSim v0.63 formal" in log.read_text(), "not a formal Exp08 output")
    summary, adaptive = load(root)
    plt.rcParams.update({"font.family": "serif", "font.size": 11, "axes.labelsize": 12, "axes.titlesize": 12, "legend.fontsize": 10})
    outputs = [metric_plot(root, summary, *item) for item in METRICS]
    outputs += [z_plot(root, adaptive), stacked(root, adaptive, "fanout"), stacked(root, adaptive, "mode")]
    for output in outputs:
        require(output.is_file() and output.stat().st_size > 0, f"missing/empty figure: {output}")
        print(f"Generated figure: {output}")


if __name__ == "__main__": main()
