from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
from scipy.stats import t


EXP07_METRICS = [
    "delivery_ratio",
    "propagation_delay",
    "duplicates",
    "total_forwards",
]


def summarize_exp07(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (strategy, fanout), group in df.groupby(["strategy", "fanout"], dropna=False):
        for metric in EXP07_METRICS:
            values = group[metric].dropna()
            n = len(values)
            mean = values.mean()
            std = values.std(ddof=1)
            half_width = t.ppf(0.975, n - 1) * std / n**0.5
            rows.append({
                "strategy": strategy,
                "fanout": fanout,
                "metric": metric,
                "n": n,
                "mean": mean,
                "std": std,
                "ci95_low": mean - half_width,
                "ci95_high": mean + half_width,
            })
    return pd.DataFrame(rows).sort_values(["strategy", "fanout", "metric"], na_position="first")


def main(path: str, output: str | None = None) -> None:
    df = pd.read_csv(path)
    experiments = set(df["experiment"].dropna().unique()) if "experiment" in df else set()
    if experiments == {"exp07"}:
        summary = summarize_exp07(df)
        if output:
            output_path = Path(output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            summary.to_csv(output_path, index=False)
            print(f"Saved {output_path}")
        print(summary.to_string(index=False))
        return

    group_cols = [c for c in ["experiment", "strategy", "fanout", "edge_prob", "ch_overload_factor"] if c in df.columns]
    summary = (
        df.groupby(group_cols, dropna=False)
        .agg(
            delivery_ratio_mean=("delivery_ratio", "mean"),
            propagation_delay_mean=("propagation_delay", "mean"),
            duplicates_mean=("duplicates", "mean"),
            total_forwards_mean=("total_forwards", "mean"),
        )
        .reset_index()
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) not in {2, 3}:
        print("Usage: python scripts/summarize_results.py RESULTS_CSV [OUTPUT_CSV]")
        raise SystemExit(1)
    main(sys.argv[1], sys.argv[2] if len(sys.argv) == 3 else None)
