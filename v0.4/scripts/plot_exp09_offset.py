from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def detect_xlabel(df: pd.DataFrame) -> str:
    topo = df["topology_type"].iloc[0]
    if topo == "er":
        return "ER Edge Probability"
    elif topo == "ba":
        return "BA Attachment Parameter (m)"
    else:
        return "Topology Parameter"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to exp09 CSV")
    parser.add_argument("--output-dir", default="outputs/plots")
    parser.add_argument("--offset", type=float, default=0.002)
    parser.add_argument("--show-values", action="store_true")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    df = df[df["strategy"].isin(["gossip", "cluster", "ahbn"])].copy()

    grouped = (
        df.groupby(["strategy", "topology_param"], as_index=False)
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delay_std=("propagation_delay", "std"),
            dup_mean=("duplicates", "mean"),
            dup_std=("duplicates", "std"),
        )
        .sort_values(["strategy", "topology_param"])
    )

    gossip = grouped[grouped["strategy"] == "gossip"]
    cluster = grouped[grouped["strategy"] == "cluster"]
    ahbn = grouped[grouped["strategy"] == "ahbn"]

    if args.show_values:
        print("\nGossip:")
        print(gossip)
        print("\nCluster:")
        print(cluster)
        print("\nAHBN:")
        print(ahbn)

    xlabel = detect_xlabel(df)
    x_ticks = sorted(df["topology_param"].unique())

    offset = args.offset
    gossip_x = gossip["topology_param"] - offset
    cluster_x = cluster["topology_param"]
    ahbn_x = ahbn["topology_param"] + offset

    output_png = output_dir / f"{csv_path.stem}_offset_plot.png"
    output_pdf = output_dir / f"{csv_path.stem}_offset_plot.pdf"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    # Delay
    ax = axes[0]
    ax.errorbar(gossip_x, gossip["delay_mean"], yerr=gossip["delay_std"].fillna(0),
                fmt="o--", capsize=3, zorder=5, label="Gossip")
    ax.errorbar(cluster_x, cluster["delay_mean"], yerr=cluster["delay_std"].fillna(0),
                fmt="d:", capsize=3, zorder=4, label="Cluster")
    ax.errorbar(ahbn_x, ahbn["delay_mean"], yerr=ahbn["delay_std"].fillna(0),
                fmt="s-", capsize=3, zorder=3, label="AHBN")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Propagation Delay")
    ax.set_title("(a) Delay vs Topology Parameter")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # Duplicates
    ax = axes[1]
    ax.errorbar(gossip_x, gossip["dup_mean"], yerr=gossip["dup_std"].fillna(0),
                fmt="o--", capsize=3, zorder=5, label="Gossip")
    ax.errorbar(cluster_x, cluster["dup_mean"], yerr=cluster["dup_std"].fillna(0),
                fmt="d:", capsize=3, zorder=4, label="Cluster")
    ax.errorbar(ahbn_x, ahbn["dup_mean"], yerr=ahbn["dup_std"].fillna(0),
                fmt="s-", capsize=3, zorder=3, label="AHBN")

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Duplicates")
    ax.set_title("(b) Duplicates vs Topology Parameter")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.suptitle(f"{csv_path.stem}: Topology Sweep (ER or BA)", y=1.03)
    fig.tight_layout()

    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()