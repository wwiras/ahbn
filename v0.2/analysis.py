import pandas as pd
import matplotlib.pyplot as plt
import os


def make_summary(df):

    group_cols = ["protocol", "node_count", "fanout", "ch_count", "ba_m", "scenario"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg({
            "propagation_time": ["mean", "std"],
            "duplicate_messages": ["mean", "std"],
            "total_transmissions": ["mean", "std"],
            "delivery_ratio": ["mean", "std"]
        })
        .reset_index()
    )

    summary.columns = ['_'.join(col).strip('_') for col in summary.columns.values]

    return summary


def plot_line(df, x, y, title, filename):
    plt.figure()

    for protocol, group in df.groupby("protocol"):
        group = group.sort_values(x)
        plt.plot(group[x], group[y], marker="o", label=protocol)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def plot_tradeoff(df, x, y, title, filename):
    plt.figure()

    for protocol, group in df.groupby("protocol"):
        plt.plot(group[x], group[y], marker="o", linestyle="-", label=protocol)

    plt.xlabel(x)
    plt.ylabel(y)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=220)
    plt.close()


def generate_figures(exp_id, df):

    os.makedirs("results/figures", exist_ok=True)

    # fanout experiments
    if "fanout" in df.columns and df["fanout"].notna().any():
        grouped = df.groupby(["protocol", "fanout"], dropna=False).mean(numeric_only=True).reset_index()

        plot_line(
            grouped,
            "fanout",
            "duplicate_messages",
            "Duplicate Messages vs Fanout",
            f"results/figures/{exp_id}_duplicates_vs_fanout.png"
        )

        plot_line(
            grouped,
            "fanout",
            "propagation_time",
            "Propagation Time vs Fanout",
            f"results/figures/{exp_id}_delay_vs_fanout.png"
        )

        plot_tradeoff(
            grouped,
            "propagation_time",
            "duplicate_messages",
            "Delay vs Duplicate Trade-off",
            f"results/figures/{exp_id}_tradeoff_delay_vs_duplicates.png"
        )

    # topology density experiments
    if "ba_m" in df.columns and df["ba_m"].notna().any() and df["ba_m"].nunique() > 1:
        grouped = df.groupby(["protocol", "ba_m"], dropna=False).mean(numeric_only=True).reset_index()

        plot_line(
            grouped,
            "ba_m",
            "duplicate_messages",
            "Duplicate Messages vs Topology Density",
            f"results/figures/{exp_id}_duplicates_vs_bam.png"
        )

        plot_line(
            grouped,
            "ba_m",
            "propagation_time",
            "Propagation Time vs Topology Density",
            f"results/figures/{exp_id}_delay_vs_bam.png"
        )

    # CH count experiments
    if "ch_count" in df.columns and df["ch_count"].notna().any():
        grouped = df.groupby(["protocol", "ch_count"], dropna=False).mean(numeric_only=True).reset_index()

        plot_line(
            grouped,
            "ch_count",
            "propagation_time",
            "Propagation Time vs CH Count",
            f"results/figures/{exp_id}_delay_vs_chcount.png"
        )

    # scenario experiments
    if "scenario" in df.columns and df["scenario"].nunique() > 1:
        grouped = df.groupby(["protocol", "scenario"], dropna=False).mean(numeric_only=True).reset_index()

        plt.figure()
        for protocol, group in grouped.groupby("protocol"):
            plt.plot(group["scenario"], group["propagation_time"], marker="o", label=protocol)
        plt.xlabel("scenario")
        plt.ylabel("propagation_time")
        plt.title("Propagation Time by Scenario")
        plt.legend()
        plt.tight_layout()
        plt.savefig(f"results/figures/{exp_id}_delay_by_scenario.png", dpi=220)
        plt.close()