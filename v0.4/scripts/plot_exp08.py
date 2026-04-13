from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to exp08 CSV")
    parser.add_argument("--output-dir", default="outputs/plots")
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    required_cols = {
        "strategy", "ch_overload_factor", "propagation_delay", "duplicates"
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {sorted(missing)}")

    df = df[df["strategy"].isin(["cluster", "ahbn"])].copy()
    if df.empty:
        raise ValueError("No cluster/AHBN rows found.")

    grouped = (
        df.groupby(["strategy", "ch_overload_factor"], as_index=False)
        .agg(
            delay_mean=("propagation_delay", "mean"),
            delay_std=("propagation_delay", "std"),
            dup_mean=("duplicates", "mean"),
            dup_std=("duplicates", "std"),
        )
        .sort_values(["strategy", "ch_overload_factor"])
    )

    cluster = grouped[grouped["strategy"] == "cluster"].sort_values("ch_overload_factor")
    ahbn = grouped[grouped["strategy"] == "ahbn"].sort_values("ch_overload_factor")

    output_png = output_dir / f"{csv_path.stem}_plot.png"
    output_pdf = output_dir / f"{csv_path.stem}_plot.pdf"

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
    x_ticks = sorted(df["ch_overload_factor"].dropna().unique())

    # Delay
    ax = axes[0]
    ax.errorbar(cluster["ch_overload_factor"], cluster["delay_mean"], yerr=cluster["delay_std"].fillna(0),
                fmt="o--", capsize=3, label="Cluster")
    ax.errorbar(ahbn["ch_overload_factor"], ahbn["delay_mean"], yerr=ahbn["delay_std"].fillna(0),
                fmt="s-", capsize=3, label="AHBN")
    ax.set_xlabel("CH Overload Factor")
    ax.set_ylabel("Average Propagation Delay")
    ax.set_title("(a) Delay vs CH Overload Factor")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True)

    # Duplicates
    ax = axes[1]
    ax.errorbar(cluster["ch_overload_factor"], cluster["dup_mean"], yerr=cluster["dup_std"].fillna(0),
                fmt="o--", capsize=3, label="Cluster")
    ax.errorbar(ahbn["ch_overload_factor"], ahbn["dup_mean"], yerr=ahbn["dup_std"].fillna(0),
                fmt="s-", capsize=3, label="AHBN")
    ax.set_xlabel("CH Overload Factor")
    ax.set_ylabel("Average Duplicates")
    ax.set_title("(b) Duplicates vs CH Overload Factor")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True)

    fig.suptitle(f"{csv_path.stem}: Cluster vs AHBN", y=1.03, fontsize=13)
    fig.tight_layout()
    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()