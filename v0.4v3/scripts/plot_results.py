from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running:
# python scripts/plot_results.py ...
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from ahbn.utils import ensure_dir, extract_timestamp_from_filename


def get_timestamp_from_csv_path(csv_path: str) -> str:
    ts = extract_timestamp_from_filename(csv_path)
    if ts is None:
        raise ValueError(
            f"Could not extract timestamp from CSV filename: {csv_path}\n"
            "Expected something like: exp07_results_20260402_093015.csv"
        )
    return ts


def plot_exp07(df: pd.DataFrame, timestamp: str) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "fanout"], dropna=False)
        .agg(
            duplicates_mean=("duplicates", "mean"),
            delay_mean=("propagation_delay", "mean"),
        )
        .reset_index()
    )

    for strategy in grouped["strategy"].unique():
        part = grouped[grouped["strategy"] == strategy].sort_values("fanout")

        plt.figure()
        plt.plot(part["fanout"], part["delay_mean"], marker="o")
        plt.xlabel("Fanout")
        plt.ylabel("Average Propagation Delay")
        plt.title(f"Exp07 - {strategy} delay vs fanout")
        plt.savefig(
            f"outputs/plots/exp07_{strategy}_delay_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        plt.plot(part["fanout"], part["duplicates_mean"], marker="o")
        plt.xlabel("Fanout")
        plt.ylabel("Average Duplicates")
        plt.title(f"Exp07 - {strategy} duplicates vs fanout")
        plt.savefig(
            f"outputs/plots/exp07_{strategy}_duplicates_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()


# def plot_exp08(df: pd.DataFrame, timestamp: str) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "ch_overload_factor"], dropna=False)
        .agg(
            delivery_ratio_mean=("delivery_ratio", "mean"),
            delay_mean=("propagation_delay", "mean"),
            duplicates_mean=("duplicates", "mean"),
        )
        .reset_index()
    )

    for strategy in grouped["strategy"].unique():
        part = grouped[grouped["strategy"] == strategy].sort_values("ch_overload_factor")

        plt.figure()
        plt.plot(part["ch_overload_factor"], part["delay_mean"], marker="o")
        plt.xlabel("CH Overload Factor")
        plt.ylabel("Average Propagation Delay")
        plt.title(f"Exp08 - {strategy} delay vs CH overload")
        plt.savefig(
            f"outputs/plots/exp08_{strategy}_delay_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        plt.plot(part["ch_overload_factor"], part["duplicates_mean"], marker="o")
        plt.xlabel("CH Overload Factor")
        plt.ylabel("Average Duplicates")
        plt.title(f"Exp08 - {strategy} duplicates vs CH overload")
        plt.savefig(
            f"outputs/plots/exp08_{strategy}_duplicates_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        plt.plot(part["ch_overload_factor"], part["delivery_ratio_mean"], marker="o")
        plt.xlabel("CH Overload Factor")
        plt.ylabel("Average Delivery Ratio")
        plt.title(f"Exp08 - {strategy} delivery vs CH overload")
        plt.savefig(
            f"outputs/plots/exp08_{strategy}_delivery_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

def plot_exp08(df, timestamp):
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "ch_overload_factor"])
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delivery_mean=("delivery_ratio", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Delay subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        axes[0].plot(part["ch_overload_factor"], part["delay_mean"], marker="o", label=s)

    axes[0].set_xlabel("CH Overload Factor")
    axes[0].set_ylabel("Propagation Delay")
    axes[0].set_title("Delay vs CH Overload")
    axes[0].legend()

    # Delivery subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("ch_overload_factor")
        axes[1].plot(part["ch_overload_factor"], part["delivery_mean"], marker="o", label=s)

    axes[1].set_xlabel("CH Overload Factor")
    axes[1].set_ylabel("Delivery Ratio")
    axes[1].set_title("Delivery vs CH Overload")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"outputs/plots/exp08_combined_{timestamp}.png")
    plt.close()


# def plot_exp09(df: pd.DataFrame, timestamp: str) -> None:
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "topology_param"], dropna=False)
        .agg(
            duplicates_mean=("duplicates", "mean"),
            delay_mean=("propagation_delay", "mean"),
            delivery_ratio_mean=("delivery_ratio", "mean"),
        )
        .reset_index()
    )

    for strategy in grouped["strategy"].unique():
        part = grouped[grouped["strategy"] == strategy].sort_values("topology_param")

        plt.figure()
        plt.plot(part["topology_param"], part["delay_mean"], marker="o")
        plt.xlabel("Edge Probability")
        plt.ylabel("Average Propagation Delay")
        plt.title(f"Exp09 - {strategy} delay vs density")
        plt.savefig(
            f"outputs/plots/exp09_{strategy}_delay_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        plt.plot(part["topology_param"], part["duplicates_mean"], marker="o")
        plt.xlabel("Edge Probability")
        plt.ylabel("Average Duplicates")
        plt.title(f"Exp09 - {strategy} duplicates vs density")
        plt.savefig(
            f"outputs/plots/exp09_{strategy}_duplicates_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

        plt.figure()
        plt.plot(part["topology_param"], part["delivery_ratio_mean"], marker="o")
        plt.xlabel("Edge Probability")
        plt.ylabel("Average Delivery Ratio")
        plt.title(f"Exp09 - {strategy} delivery vs density")
        plt.savefig(
            f"outputs/plots/exp09_{strategy}_delivery_{timestamp}.png",
            bbox_inches="tight",
        )
        plt.close()

def plot_exp09(df, timestamp):
    ensure_dir("outputs/plots")

    grouped = (
        df.groupby(["strategy", "topology_param"])
        .agg(
            duplicates_mean=("duplicates", "mean"),
            delay_mean=("propagation_delay", "mean"),
        )
        .reset_index()
    )

    strategies = grouped["strategy"].unique()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Duplicates subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        axes[0].plot(part["topology_param"], part["duplicates_mean"], marker="o", label=s)

    axes[0].set_xlabel("Edge Probability")
    axes[0].set_ylabel("Duplicates")
    axes[0].set_title("Duplicates vs Density")
    axes[0].legend()

    # Delay subplot
    for s in strategies:
        part = grouped[grouped["strategy"] == s].sort_values("topology_param")
        axes[1].plot(part["topology_param"], part["delay_mean"], marker="o", label=s)

    axes[1].set_xlabel("Edge Probability")
    axes[1].set_ylabel("Propagation Delay")
    axes[1].set_title("Delay vs Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig(f"outputs/plots/exp09_combined_{timestamp}.png")
    plt.close()

def main(path: str) -> None:
    df = pd.read_csv(path)
    timestamp = get_timestamp_from_csv_path(path)

    experiment = df["experiment"].iloc[0]

    if experiment == "exp07":
        plot_exp07(df, timestamp)
    elif experiment == "exp08":
        plot_exp08(df, timestamp)
    elif experiment == "exp09":
        plot_exp09(df, timestamp)
    else:
        raise ValueError(f"Unknown experiment: {experiment}")

    print(f"Plots saved to outputs/plots/ with timestamp: {timestamp}")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_results.py outputs/csv/exp07_results_20260402_093015.csv")
        raise SystemExit(1)

    main(sys.argv[1])