import pandas as pd
import matplotlib.pyplot as plt
import os

def make_summary(df):

    group_cols = ["protocol","node_count","fanout","ch_count","ba_m","scenario"]

    summary = (
        df.groupby(group_cols, dropna=False)
        .agg({
            "propagation_time":["mean","std"],
            "duplicate_messages":["mean","std"],
            "total_transmissions":["mean","std"]
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
    plt.savefig(filename)
    plt.close()


def generate_figures(exp_id, df):

    os.makedirs("results/figures", exist_ok=True)

    if exp_id.startswith("exp01"):

        grouped = df.groupby(["protocol","fanout"]).mean(numeric_only=True).reset_index()

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

    if exp_id.startswith("exp03"):

        grouped = df.groupby(["protocol","ba_m"]).mean(numeric_only=True).reset_index()

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