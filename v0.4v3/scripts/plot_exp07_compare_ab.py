from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_grouped(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    required_cols = {"strategy", "fanout", "propagation_delay", "duplicates"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path.name} missing columns: {sorted(missing)}")

    df = df[df["strategy"].isin(["gossip", "ahbn"])].copy()
    if df.empty:
        raise ValueError(f"No gossip/AHBN rows in {csv_path.name}")

    grouped = (
        df.groupby(["strategy", "fanout"], as_index=False)
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delay_std=("propagation_delay", "std"),
            dup_mean=("duplicates", "mean"),
            dup_std=("duplicates", "std"),
        )
        .sort_values(["strategy", "fanout"])
    )
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_a", help="Path to exp07a CSV")
    parser.add_argument("csv_b", help="Path to exp07b CSV")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    csv_a = Path(args.csv_a)
    csv_b = Path(args.csv_b)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    grouped_a = load_grouped(csv_a)
    grouped_b = load_grouped(csv_b)

    gossip = grouped_a[grouped_a["strategy"] == "gossip"].sort_values("fanout")
    ahbn_a = grouped_a[grouped_a["strategy"] == "ahbn"].sort_values("fanout")
    ahbn_b = grouped_b[grouped_b["strategy"] == "ahbn"].sort_values("fanout")

    output_png = output_dir / f"exp07ab_comparison_{csv_a.stem}_{csv_b.stem}.png"
    output_pdf = output_dir / f"exp07ab_comparison_{csv_a.stem}_{csv_b.stem}.pdf"

    plt.rcParams.update({
        "font.family": "serif",
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.linewidth": 1.0,
        "lines.linewidth": 2.0,
        "lines.markersize": 6,
    })

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))
    x_ticks = sorted(gossip["fanout"].dropna().unique())

    # Delay
    ax = axes[0]
    ax.errorbar(gossip["fanout"], gossip["delay_mean"], yerr=gossip["delay_std"].fillna(0),
                fmt="o--", capsize=3, label="Gossip")
    ax.errorbar(ahbn_a["fanout"], ahbn_a["delay_mean"], yerr=ahbn_a["delay_std"].fillna(0),
                fmt="s-", capsize=3, label="AHBN-A")
    ax.errorbar(ahbn_b["fanout"], ahbn_b["delay_mean"], yerr=ahbn_b["delay_std"].fillna(0),
                fmt="^-.", capsize=3, label="AHBN-B")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Average Propagation Delay")
    ax.set_title("(a) Delay vs Fanout")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True)

    # Duplicates
    ax = axes[1]
    ax.errorbar(gossip["fanout"], gossip["dup_mean"], yerr=gossip["dup_std"].fillna(0),
                fmt="o--", capsize=3, label="Gossip")
    ax.errorbar(ahbn_a["fanout"], ahbn_a["dup_mean"], yerr=ahbn_a["dup_std"].fillna(0),
                fmt="s-", capsize=3, label="AHBN-A")
    ax.errorbar(ahbn_b["fanout"], ahbn_b["dup_mean"], yerr=ahbn_b["dup_std"].fillna(0),
                fmt="^-.", capsize=3, label="AHBN-B")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Average Duplicates")
    ax.set_title("(b) Duplicates vs Fanout")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True)

    fig.suptitle("Experiment 07 Comparison: Gossip vs AHBN-A vs AHBN-B", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()