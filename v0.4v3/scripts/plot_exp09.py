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
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print aggregated values",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)

    required_cols = {
        "strategy",
        "topology_param",
        "propagation_delay",
        "duplicates",
        "topology_type",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df[df["strategy"].isin(["gossip", "cluster", "ahbn"])].copy()
    if df.empty:
        raise ValueError("No valid strategies found.")

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
        print(gossip.to_string(index=False))
        print("\nCluster:")
        print(cluster.to_string(index=False))
        print("\nAHBN:")
        print(ahbn.to_string(index=False))

    xlabel = detect_xlabel(df)
    x_ticks = sorted(df["topology_param"].unique())

    output_png = output_dir / f"{csv_path.stem}_plot.png"
    output_pdf = output_dir / f"{csv_path.stem}_plot.pdf"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.8))

    # -------- Delay --------
    ax = axes[0]

    ax.errorbar(
        gossip["topology_param"],
        gossip["delay_mean"],
        yerr=gossip["delay_std"].fillna(0),
        fmt="o--",
        capsize=3,
        label="Gossip",
    )

    ax.errorbar(
        cluster["topology_param"],
        cluster["delay_mean"],
        yerr=cluster["delay_std"].fillna(0),
        fmt="d:",
        capsize=3,
        label="Cluster",
    )

    ax.errorbar(
        ahbn["topology_param"],
        ahbn["delay_mean"],
        yerr=ahbn["delay_std"].fillna(0),
        fmt="s-",
        capsize=3,
        label="AHBN",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Propagation Delay")
    ax.set_title("(a) Delay vs Topology Parameter")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    # -------- Duplicates --------
    ax = axes[1]

    ax.errorbar(
        gossip["topology_param"],
        gossip["dup_mean"],
        yerr=gossip["dup_std"].fillna(0),
        fmt="o--",
        capsize=3,
        label="Gossip",
    )

    ax.errorbar(
        cluster["topology_param"],
        cluster["dup_mean"],
        yerr=cluster["dup_std"].fillna(0),
        fmt="d:",
        capsize=3,
        label="Cluster",
    )

    ax.errorbar(
        ahbn["topology_param"],
        ahbn["dup_mean"],
        yerr=ahbn["dup_std"].fillna(0),
        fmt="s-",
        capsize=3,
        label="AHBN",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel("Average Duplicates")
    ax.set_title("(b) Duplicates vs Topology Parameter")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend()

    fig.suptitle(f"{csv_path.stem}: Gossip vs Cluster vs AHBN", y=1.03)
    fig.tight_layout()

    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()