from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import argparse
import matplotlib.pyplot as plt
import pandas as pd

from ahbn.utils import ensure_dir, extract_timestamp_from_filename


# -----------------------------
# Helpers
# -----------------------------
def get_timestamp(csv_path: str) -> str:
    ts = extract_timestamp_from_filename(csv_path)
    if ts is None:
        raise ValueError(f"Cannot extract timestamp from {csv_path}")
    return ts


def detect_xlabel(df: pd.DataFrame) -> str:
    topo = df["topology_type"].iloc[0]
    if topo == "er":
        return "ER Edge Probability"
    elif topo == "ba":
        return "BA Attachment Parameter (m)"
    return "Topology Parameter"


def apply_offset(series: pd.Series, offset: float) -> pd.Series:
    return series.astype(float) + offset


def get_plot_output_path(experiment: str, timestamp: str) -> str:
    return f"outputs/plots/{experiment}_combined_{timestamp}.png"


# -----------------------------
# Exp07
# -----------------------------
def plot_exp07(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    df = df[df["strategy"].isin(["gossip", "ahbn"])].copy()

    grouped = (
        df.groupby(["strategy", "fanout"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            dup_mean=("duplicates", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()
    offsets = {"gossip": -0.05, "ahbn": 0.05} if use_offset else {s: 0.0 for s in strategies}

    x_ticks = sorted(df["fanout"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Delay subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("fanout")
        x = apply_offset(part["fanout"], offsets[s])
        axes[0].plot(x, part["delay_mean"], marker="o", label=s)

    axes[0].set_xlabel("Fanout")
    axes[0].set_ylabel("Propagation Delay")
    axes[0].set_title("Delay vs Fanout")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    # Duplicates subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("fanout")
        x = apply_offset(part["fanout"], offsets[s])
        axes[1].plot(x, part["dup_mean"], marker="o", label=s)

    axes[1].set_xlabel("Fanout")
    axes[1].set_ylabel("Duplicates")
    axes[1].set_title("Duplicates vs Fanout")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp07", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp08
# -----------------------------
def plot_exp08(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    df = df[df["strategy"].isin(["cluster", "ahbn"])].copy()

    grouped = (
        df.groupby(["strategy", "ch_overload_factor"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()
    offsets = {"cluster": -0.03, "ahbn": 0.03} if use_offset else {s: 0.0 for s in strategies}

    x_ticks = sorted(df["ch_overload_factor"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.2))

    # Delay subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        x = apply_offset(part["ch_overload_factor"], offsets[s])
        axes[0].plot(x, part["delay_mean"], marker="o", label=s)

    axes[0].set_xlabel("CH Overload Factor")
    axes[0].set_ylabel("Delay")
    axes[0].set_title("Delay vs CH Overload")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    # Delivery subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        x = apply_offset(part["ch_overload_factor"], offsets[s])
        axes[1].plot(x, part["delivery_mean"], marker="o", label=s)

    axes[1].set_xlabel("CH Overload Factor")
    axes[1].set_ylabel("Delivery Ratio")
    axes[1].set_title("Delivery vs CH Overload")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp08", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")


# -----------------------------
# Exp09
# -----------------------------
def plot_exp09(df: pd.DataFrame, ts: str, use_offset: bool) -> None:
    ensure_dir("outputs/plots")

    xlabel = detect_xlabel(df)

    grouped = (
        df.groupby(["strategy", "topology_param"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            dup_mean=("duplicates", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()

    if use_offset:
        offsets = {"gossip": -0.002, "cluster": 0.0, "ahbn": 0.002}
    else:
        offsets = {s: 0.0 for s in strategies}

    x_ticks = sorted(df["topology_param"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.2))

    # Duplicates subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        x = apply_offset(part["topology_param"], offsets[s])
        axes[0].plot(x, part["dup_mean"], marker="o", label=s)

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel("Duplicates")
    axes[0].set_title("Duplicates vs Topology")
    axes[0].set_xticks(x_ticks)
    axes[0].legend()
    axes[0].grid(True, linestyle=":")

    # Delay subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        x = apply_offset(part["topology_param"], offsets[s])
        axes[1].plot(x, part["delay_mean"], marker="o", label=s)

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel("Delay")
    axes[1].set_title("Delay vs Topology")
    axes[1].set_xticks(x_ticks)
    axes[1].legend()
    axes[1].grid(True, linestyle=":")

    plt.tight_layout()
    out = get_plot_output_path("exp09", ts)
    plt.savefig(out, bbox_inches="tight")
    plt.close()

    print(f"Saved {out}")
    
    
    


# -----------------------------
# Main
# -----------------------------
def main(path: str, use_offset: bool) -> None:
    df = pd.read_csv(path)
    ts = get_timestamp(path)

    experiment = df["experiment"].iloc[0]
    df = df[df["experiment"] == experiment].copy()

    if experiment == "exp07":
        plot_exp07(df, ts, use_offset)
    elif experiment == "exp08":
        plot_exp08(df, ts, use_offset)
    elif experiment == "exp09":
        plot_exp09(df, ts, use_offset)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print(f"Plots saved (offset={use_offset}) with timestamp: {ts}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path")
    parser.add_argument("--offset", action="store_true", help="Enable slight x-offset to separate overlapping curves")
    args = parser.parse_args()

    main(args.csv_path, args.offset)