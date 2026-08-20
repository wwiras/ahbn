#!/usr/bin/env python3
"""Stage 4 Exp08 E8: plot four algorithms from the frozen E7 summary."""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "outputs/csv/exp08_summary_20260820_123734.csv"
OUTPUT_PNG = ROOT / "outputs/figures/exp08_four_algorithms_e8.png"
OUTPUT_PDF = ROOT / "outputs/figures/exp08_four_algorithms_e8.pdf"

STRATEGIES = {
    "gossip": "Gossip",
    "cluster": "Structured",
    "dcsoc": "DC-SoC",
    "ahbn": "AHBN",
}
OVERLOADS = [1.0, 1.5, 2.0, 3.0]
METRICS = {
    "delivery_ratio": ("(a) Delivery ratio", "Delivery ratio"),
    "propagation_delay": ("(b) Propagation delay", "Propagation delay (s)"),
    "duplicates": ("(c) Duplicates", "Duplicates"),
    "total_forwards": ("(d) Total forwards", "Total forwards"),
}
STYLES = {
    "gossip": {"marker": "o", "linestyle": "-"},
    "cluster": {"marker": "s", "linestyle": "--"},
    "dcsoc": {"marker": "^", "linestyle": "-."},
    "ahbn": {"marker": "D", "linestyle": ":"},
}


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def validate_summary() -> pd.DataFrame:
    require(SUMMARY.is_file(), f"frozen E7 summary does not exist: {SUMMARY}")
    frame = pd.read_csv(SUMMARY)
    required = {"strategy", "ch_overload_factor"}
    for metric in METRICS:
        required.update(
            {f"{metric}_mean", f"{metric}_ci95_low", f"{metric}_ci95_high"}
        )
    missing = sorted(required - set(frame.columns))
    require(not missing, f"missing required E7 summary columns: {missing}")
    require(len(frame) == 16, f"expected 16 aggregated conditions, found {len(frame)}")
    require(
        set(frame["strategy"]) == set(STRATEGIES),
        f"expected strategies {sorted(STRATEGIES)}, found {sorted(frame['strategy'].unique())}",
    )

    overload = pd.to_numeric(frame["ch_overload_factor"], errors="coerce")
    require(overload.notna().all(), "ch_overload_factor contains non-numeric values")
    frame["ch_overload_factor"] = overload.astype(float)
    require(
        set(frame["ch_overload_factor"]) == set(OVERLOADS),
        f"expected overload factors {OVERLOADS}, found {sorted(frame['ch_overload_factor'].unique())}",
    )
    require(
        not frame.duplicated(["strategy", "ch_overload_factor"]).any(),
        "duplicate strategy x overload-factor condition found",
    )
    expected = {(strategy, overload) for strategy in STRATEGIES for overload in OVERLOADS}
    actual = set(zip(frame["strategy"], frame["ch_overload_factor"]))
    require(actual == expected, "strategy x overload-factor grid is incomplete or has extras")

    if "n" in frame.columns:
        n = pd.to_numeric(frame["n"], errors="coerce")
        require(n.notna().all() and (n == 20).all(), "expected n=20 for every condition")

    for metric in METRICS:
        mean_col = f"{metric}_mean"
        low_col = f"{metric}_ci95_low"
        high_col = f"{metric}_ci95_high"
        for column in (mean_col, low_col, high_col):
            values = pd.to_numeric(frame[column], errors="coerce")
            require(
                values.notna().all() and values.map(math.isfinite).all(),
                f"{column} contains non-numeric or non-finite values",
            )
            frame[column] = values
        require(
            ((frame[low_col] <= frame[mean_col]) &
             (frame[mean_col] <= frame[high_col])).all(),
            f"{metric}: one or more 95% CIs do not contain the mean",
        )
    return frame


def main() -> None:
    print("E8 Exp08 plotting")
    print(f"Numerical input (frozen E7 summary only): {SUMMARY.relative_to(ROOT)}")
    print(f"Exact E7 columns: {list(pd.read_csv(SUMMARY, nrows=0).columns)}")
    existed = {OUTPUT_PNG: OUTPUT_PNG.exists(), OUTPUT_PDF: OUTPUT_PDF.exists()}
    for path, was_present in existed.items():
        print(f"Pre-existing output {path.relative_to(ROOT)}: {'YES (will replace)' if was_present else 'NO'}")

    frame = validate_summary()
    print("Validation: PASS")
    print(f"Conditions: {len(frame)}")
    print(f"Strategies found: {sorted(frame['strategy'].unique())}")
    print(f"Overload factors found: {sorted(frame['ch_overload_factor'].unique())}")
    print(f"Runs per condition: {sorted(frame['n'].unique()) if 'n' in frame else 'not present'}")

    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][:4]
    point_counts = {}
    for ax, (metric, (title, ylabel)) in zip(axes.flat, METRICS.items()):
        count = 0
        for color, (strategy, display_name) in zip(colors, STRATEGIES.items()):
            rows = frame.loc[frame["strategy"] == strategy].sort_values(
                "ch_overload_factor"
            )
            require(len(rows) == 4, f"{strategy}/{metric}: expected 4 rows, found {len(rows)}")
            x = rows["ch_overload_factor"].to_numpy()
            mean = rows[f"{metric}_mean"].to_numpy()
            low = rows[f"{metric}_ci95_low"].to_numpy()
            high = rows[f"{metric}_ci95_high"].to_numpy()
            ax.errorbar(
                x,
                mean,
                yerr=[mean - low, high - mean],
                label=display_name,
                color=color,
                linewidth=1.6,
                markersize=5.5,
                capsize=3,
                elinewidth=1.1,
                **STYLES[strategy],
            )
            count += len(rows)
        point_counts[metric] = count
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("CH overload factor")
        ax.set_ylabel(ylabel)
        ax.set_xticks(OVERLOADS)
        ax.grid(True, linestyle=":", linewidth=0.7, alpha=0.65)

    require(
        point_counts == {metric: 16 for metric in METRICS},
        f"plotted-point integrity failure: {point_counts}",
    )
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside upper center", ncols=4, frameon=False)
    fig.suptitle("Exp08: Four-algorithm comparison", fontsize=14)
    OUTPUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(fig)

    require(OUTPUT_PNG.is_file() and OUTPUT_PNG.stat().st_size > 0, "PNG output missing or empty")
    require(OUTPUT_PDF.is_file() and OUTPUT_PDF.stat().st_size > 0, "PDF output missing or empty")
    print(f"Metrics plotted: {list(METRICS)}")
    for metric, count in point_counts.items():
        print(f"Integrity {metric}: {count} E7 means + {count} corresponding E7 95% CIs")
    print(f"Saved: {OUTPUT_PNG.relative_to(ROOT)}")
    print(f"Saved: {OUTPUT_PDF.relative_to(ROOT)}")
    print("No raw CSV or adaptive trace was used as numerical input.")
    print("No simulation or aggregation was run.")
    print("Overall E8: PASS")


if __name__ == "__main__":
    main()
