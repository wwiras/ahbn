from __future__ import annotations

import sys
from pathlib import Path

# Make project root importable when running:
# python scripts/plot_results.py outputs/csv/exp07_results_20260413_224800.csv
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import pandas as pd

from ahbn.utils import ensure_dir


def plot_exp07(csv_path: str) -> None:
    df = pd.read_csv(csv_path)

    required_cols = {
        "experiment",
        "strategy",
        "fanout",
        "delivery_ratio",
        "propagation_delay",
        "duplicates",
    }
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"CSV is missing required columns: {sorted(missing)}")

    exp_values = set(df["experiment"].dropna().unique())
    if exp_values != {"exp07"} and "exp07" not in exp_values:
        print(
            f"Warning: CSV contains experiment values {sorted(exp_values)}. "
            f"This plotting script is intended for exp07."
        )

    grouped = (
        df.groupby(["fanout", "strategy"], as_index=False)[
            ["delivery_ratio", "propagation_delay", "duplicates"]
        ]
        .mean()
        .sort_values(["fanout", "strategy"])
    )

    preferred_order = ["ahbn", "gossip", "hybrid_fixed", "cluster"]
    strategies = [s for s in preferred_order if s in grouped["strategy"].unique()]
    for s in grouped["strategy"].unique():
        if s not in strategies:
            strategies.append(s)

    csv_name = Path(csv_path).stem
    ensure_dir("outputs/plots")

    # -----------------------------
    # 3-panel figure
    # -----------------------------
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Delivery Ratio
    ax = axes[0]
    for strategy in strategies:
        subset = grouped[grouped["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["delivery_ratio"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Delivery Ratio vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Delivery Ratio")
    ax.grid(True)
    ax.legend()

    # Delay
    ax = axes[1]
    for strategy in strategies:
        subset = grouped[grouped["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["propagation_delay"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Delay vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Propagation Delay")
    ax.grid(True)
    ax.legend()

    # Duplicates
    ax = axes[2]
    for strategy in strategies:
        subset = grouped[grouped["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["duplicates"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Duplicates vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Duplicates")
    ax.grid(True)
    ax.legend()

    fig.tight_layout()

    out_3panel = f"outputs/plots/{csv_name}_3panel.png"
    fig.savefig(out_3panel, dpi=150, bbox_inches="tight")
    # plt.show()

    # -----------------------------
    # 2-panel figure
    # -----------------------------
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))

    # Delay
    ax = axes2[0]
    for strategy in strategies:
        subset = grouped[grouped["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["propagation_delay"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Delay vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Propagation Delay")
    ax.grid(True)
    ax.legend()

    # Duplicates
    ax = axes2[1]
    for strategy in strategies:
        subset = grouped[grouped["strategy"] == strategy]
        ax.plot(
            subset["fanout"],
            subset["duplicates"],
            marker="o",
            label=strategy,
        )
    ax.set_title("Duplicates vs Fanout")
    ax.set_xlabel("Fanout")
    ax.set_ylabel("Duplicates")
    ax.grid(True)
    ax.legend()

    fig2.tight_layout()

    out_2panel = f"outputs/plots/{csv_name}_2panel.png"
    fig2.savefig(out_2panel, dpi=150, bbox_inches="tight")
    # plt.show()

    print(f"Saved 3-panel plot to {out_3panel}")
    print(f"Saved 2-panel plot to {out_2panel}")


def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python scripts/plot_results.py <path_to_exp07_csv>")
        sys.exit(1)

    csv_path = sys.argv[1]
    plot_exp07(csv_path)


if __name__ == "__main__":
    main()