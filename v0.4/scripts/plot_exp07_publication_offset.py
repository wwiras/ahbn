from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def extract_timestamp_from_filename(path: Path) -> str:
    stem = path.stem
    parts = stem.split("_")
    if len(parts) >= 2:
        maybe_date = parts[-2]
        maybe_time = parts[-1]
        if maybe_date.isdigit() and maybe_time.isdigit():
            return f"{maybe_date}_{maybe_time}"
    return "latest"


def load_and_aggregate(csv_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(csv_path)

    required_cols = {
        "experiment",
        "strategy",
        "fanout",
        "propagation_delay",
        "duplicates",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    df = df[df["strategy"].isin(["gossip", "ahbn"])].copy()

    if df.empty:
        raise ValueError("No gossip/AHBN rows found in the CSV.")

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

    gossip = grouped[grouped["strategy"] == "gossip"].sort_values("fanout")
    ahbn = grouped[grouped["strategy"] == "ahbn"].sort_values("fanout")

    if gossip.empty or ahbn.empty:
        raise ValueError("Both gossip and AHBN series must be present.")

    return gossip, ahbn


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv_path", help="Path to exp07/exp07a/exp07b CSV")
    parser.add_argument(
        "--output-dir",
        default="outputs/plots",
        help="Directory to save the generated figure",
    )
    parser.add_argument(
        "--offset",
        type=float,
        default=0.04,
        help="Small x-offset used to separate overlapping curves visually",
    )
    parser.add_argument(
        "--show-values",
        action="store_true",
        help="Print aggregated mean values for gossip and AHBN to the console",
    )
    args = parser.parse_args()

    csv_path = Path(args.csv_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gossip, ahbn = load_and_aggregate(csv_path)
    timestamp = extract_timestamp_from_filename(csv_path)

    stem = csv_path.stem
    output_png = output_dir / f"{stem}_publication_offset.png"
    output_pdf = output_dir / f"{stem}_publication_offset.pdf"

    if args.show_values:
        print("\nGossip aggregated means:")
        print(gossip[["fanout", "delay_mean", "dup_mean"]].to_string(index=False))
        print("\nAHBN aggregated means:")
        print(ahbn[["fanout", "delay_mean", "dup_mean"]].to_string(index=False))

    # Publication-style settings
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

    # Apply slight x-offset so overlapping lines can be seen
    offset = args.offset
    gossip_x = gossip["fanout"].astype(float) - offset
    ahbn_x = ahbn["fanout"].astype(float) + offset
    x_ticks = sorted(gossip["fanout"].dropna().unique())

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # -----------------------------
    # (a) Delay vs Fanout
    # -----------------------------
    ax = axes[0]

    ax.errorbar(
        gossip_x,
        gossip["delay_mean"],
        yerr=gossip["delay_std"].fillna(0),
        fmt="o--",
        linewidth=2.0,
        markersize=6,
        capsize=3,
        alpha=0.9,
        zorder=4,
        label="Gossip",
    )

    ax.errorbar(
        ahbn_x,
        ahbn["delay_mean"],
        yerr=ahbn["delay_std"].fillna(0),
        fmt="s-",
        linewidth=2.0,
        markersize=6,
        capsize=3,
        alpha=0.9,
        zorder=3,
        label="AHBN",
    )

    ax.set_xlabel("Fanout")
    ax.set_ylabel("Average Propagation Delay")
    ax.set_title("(a) Delay vs Fanout")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True)

    # -----------------------------
    # (b) Duplicates vs Fanout
    # -----------------------------
    ax = axes[1]

    ax.errorbar(
        gossip_x,
        gossip["dup_mean"],
        yerr=gossip["dup_std"].fillna(0),
        fmt="o--",
        linewidth=2.0,
        markersize=6,
        capsize=3,
        alpha=0.9,
        zorder=4,
        label="Gossip",
    )

    ax.errorbar(
        ahbn_x,
        ahbn["dup_mean"],
        yerr=ahbn["dup_std"].fillna(0),
        fmt="s-",
        linewidth=2.0,
        markersize=6,
        capsize=3,
        alpha=0.9,
        zorder=3,
        label="AHBN",
    )

    ax.set_xlabel("Fanout")
    ax.set_ylabel("Average Duplicates")
    ax.set_title("(b) Duplicates vs Fanout")
    ax.set_xticks(x_ticks)
    ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.6)
    ax.legend(frameon=True)

    fig.suptitle("Experiment 07: Gossip and AHBN under Fanout Variation", y=1.03, fontsize=13)
    fig.tight_layout()

    fig.savefig(output_png, dpi=400, bbox_inches="tight")
    fig.savefig(output_pdf, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved {output_png}")
    print(f"Saved {output_pdf}")


if __name__ == "__main__":
    main()