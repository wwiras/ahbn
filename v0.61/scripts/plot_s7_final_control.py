#!/usr/bin/env python3
"""Generate S7 manuscript figures exclusively from frozen S6 statistics."""

from __future__ import annotations

import csv
import hashlib
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


ROOT = Path(__file__).resolve().parents[1]
INPUT = ROOT / "outputs/csv/final_control_statistics_s6.csv"
PLOTDATA = ROOT / "outputs/csv/final_control_plotdata_s7.csv"
FIGDIR = ROOT / "outputs/figures/s7"
EXPECTED_S6_SHA256 = "a1f259b2a8727548639aba28f6bcfebf08ebf6d0718dc18ac10c0c79d3e63ca4"

ALGORITHMS = ("Gossip", "Structured", "DC-SoC", "AHBN")
STYLE = {
    "Gossip": dict(color="#0072B2", marker="o", linestyle="-"),
    "Structured": dict(color="#E69F00", marker="s", linestyle="--"),
    "DC-SoC": dict(color="#009E73", marker="D", linestyle="-."),
    "AHBN": dict(color="#CC79A7", marker="^", linestyle=":"),
}
METRICS = {
    "delivery_ratio": "Delivery Ratio",
    "propagation_delay": "Propagation Delay (s)",
    "duplicates": "Duplicate Transmissions",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_and_validate() -> list[dict[str, str]]:
    if not INPUT.is_file():
        raise SystemExit(f"Missing authoritative S6 input: {INPUT}")
    observed = sha256(INPUT)
    if observed != EXPECTED_S6_SHA256:
        raise SystemExit(f"S6 SHA-256 mismatch: expected {EXPECTED_S6_SHA256}, observed {observed}")
    with INPUT.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    expected = {
        "Exp07": [("Gossip", f"gossip_k={k}") for k in (2, 3, 4, 5, 6)]
        + [("AHBN", "ahbn_canonical_adaptive")],
        "Exp08": [(a, f"ch_overload_factor={x}") for x in ("1.0", "1.5", "2.0", "3.0") for a in ALGORITHMS],
        "Exp09": [(a, f"edge_prob={x}") for x in ("0.04", "0.06", "0.08", "0.10", "0.12") for a in ALGORITHMS],
    }
    actual: dict[str, list[tuple[str, str]]] = {key: [] for key in expected}
    for row in rows:
        if row["experiment"] not in actual:
            raise SystemExit(f"Unexpected experiment: {row['experiment']}")
        actual[row["experiment"]].append((row["algorithm"], row["experimental_condition"]))
        if int(row["n"]) != 20:
            raise SystemExit(f"Expected n=20: {row}")
    for experiment, cells in expected.items():
        if len(actual[experiment]) != len(set(actual[experiment])):
            raise SystemExit(f"Duplicate S6 cell in {experiment}")
        missing = set(cells) - set(actual[experiment])
        extra = set(actual[experiment]) - set(cells)
        if missing or extra:
            raise SystemExit(f"Invalid {experiment} coverage; missing={sorted(missing)}, extra={sorted(extra)}")
    if len(rows) != 42:
        raise SystemExit(f"Expected 42 S6 cells, found {len(rows)}")
    return rows


def row_index(rows: list[dict[str, str]]) -> dict[tuple[str, str, str], dict[str, str]]:
    return {(r["experiment"], r["algorithm"], r["experimental_condition"]): r for r in rows}


def values(row: dict[str, str], metric: str) -> tuple[float, float, float]:
    mean = float(row[f"{metric}_mean"])
    low = float(row[f"{metric}_ci95_low"])
    high = float(row[f"{metric}_ci95_high"])
    if not all(math.isfinite(v) for v in (mean, low, high)) or not low <= mean <= high:
        raise SystemExit(f"Invalid {metric} mean/CI in {row}")
    return mean, low, high


def errorbar(ax, x, rows, metric, algorithm, *, label=None, linestyle=None, zorder=2):
    triples = [values(row, metric) for row in rows]
    means = [v[0] for v in triples]
    yerr = [[v[0] - v[1] for v in triples], [v[2] - v[0] for v in triples]]
    style = STYLE[algorithm].copy()
    if linestyle is not None:
        style["linestyle"] = linestyle
    ax.errorbar(
        x, means, yerr=yerr, label=label or algorithm, linewidth=1.6, markersize=5.2,
        markeredgecolor="white", markeredgewidth=0.55, capsize=2.8, capthick=0.9,
        elinewidth=0.9, zorder=zorder, **style,
    )


def finish(fig, axes, basename, ncol=4):
    for ax in axes:
        ax.grid(axis="y", color="#d9d9d9", linewidth=0.65, alpha=0.75)
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.margins(y=0.10)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=ncol, frameon=False,
               bbox_to_anchor=(0.5, 1.015), handlelength=2.5, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.91), w_pad=2.4)
    FIGDIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGDIR / f"{basename}.pdf", bbox_inches="tight")
    fig.savefig(FIGDIR / f"{basename}.png", dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_exp07(index):
    fig, axes = plt.subplots(1, 3, figsize=(10.8, 3.55))
    gossip = [index[("Exp07", "Gossip", f"gossip_k={k}")] for k in (2, 3, 4, 5, 6)]
    ahbn = [index[("Exp07", "AHBN", "ahbn_canonical_adaptive")]]
    for panel, (ax, metric) in enumerate(zip(axes, ("delivery_ratio", "propagation_delay", "duplicates"))):
        errorbar(ax, [2, 3, 4, 5, 6], gossip, metric, "Gossip", label="Gossip fanout sweep")
        errorbar(ax, [3], ahbn, metric, "AHBN", label="AHBN canonical adaptive operating point",
                 linestyle="none", zorder=4)
        ax.set_xlabel("Forwarding Fanout")
        ax.set_ylabel(METRICS[metric])
        ax.set_xticks([2, 3, 4, 5, 6])
        ax.set_title(f"({chr(97 + panel)}) {METRICS[metric].replace(' (s)', '')}", loc="left", fontsize=10)
    finish(fig, axes, "exp07_final_fanout", ncol=2)


def plot_multi(index, experiment, conditions, xvalues, metrics, xlabel, basename):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.75))
    for panel, (ax, metric) in enumerate(zip(axes, metrics)):
        for zorder, algorithm in enumerate(ALGORITHMS, start=2):
            rows = [index[(experiment, algorithm, condition)] for condition in conditions]
            errorbar(ax, xvalues, rows, metric, algorithm, zorder=zorder)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(METRICS[metric])
        ax.set_xticks(xvalues)
        ax.set_title(f"({chr(97 + panel)}) {METRICS[metric].replace(' (s)', '')}", loc="left", fontsize=10)
    if experiment == "Exp09":
        for ax in axes:
            ax.xaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    finish(fig, axes, basename)


def write_plotdata(rows):
    panel_metrics = {
        "Exp07": ("delivery_ratio", "propagation_delay", "duplicates"),
        "Exp08": ("propagation_delay", "delivery_ratio"),
        "Exp09": ("duplicates", "propagation_delay"),
    }
    condition_order = {
        "Exp07": {f"gossip_k={k}": k for k in (2, 3, 4, 5, 6)} | {"ahbn_canonical_adaptive": 3},
        "Exp08": {f"ch_overload_factor={x}": x for x in ("1.0", "1.5", "2.0", "3.0")},
        "Exp09": {f"edge_prob={x}": x for x in ("0.04", "0.06", "0.08", "0.10", "0.12")},
    }
    algorithm_order = {a: i for i, a in enumerate(ALGORITHMS)}
    ordered = sorted(rows, key=lambda r: (r["experiment"], float(condition_order[r["experiment"]][r["experimental_condition"]]), algorithm_order[r["algorithm"]]))
    PLOTDATA.parent.mkdir(parents=True, exist_ok=True)
    with PLOTDATA.open("w", newline="", encoding="utf-8") as handle:
        fields = ("experiment", "panel_metric", "algorithm", "condition", "n", "mean", "ci_low", "ci_high")
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in ordered:
            for metric in panel_metrics[row["experiment"]]:
                writer.writerow({
                    "experiment": row["experiment"], "panel_metric": metric,
                    "algorithm": row["algorithm"], "condition": row["experimental_condition"], "n": row["n"],
                    "mean": row[f"{metric}_mean"], "ci_low": row[f"{metric}_ci95_low"], "ci_high": row[f"{metric}_ci95_high"],
                })


def verify_plotdata(rows):
    source = row_index(rows)
    checked = 0
    with PLOTDATA.open(newline="", encoding="utf-8") as handle:
        for plotted in csv.DictReader(handle):
            original = source[(plotted["experiment"], plotted["algorithm"], plotted["condition"])]
            metric = plotted["panel_metric"]
            expected = (original["n"], original[f"{metric}_mean"], original[f"{metric}_ci95_low"], original[f"{metric}_ci95_high"])
            observed = (plotted["n"], plotted["mean"], plotted["ci_low"], plotted["ci_high"])
            if observed != expected:
                raise SystemExit(f"Plot-data mismatch: {plotted}")
            checked += 1
    if checked != 90:
        raise SystemExit(f"Expected 90 plotted points, verified {checked}")
    return checked


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.labelsize": 9.5,
        "axes.titlesize": 10, "legend.fontsize": 8.5, "xtick.labelsize": 8.5,
        "ytick.labelsize": 8.5, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    rows = read_and_validate()
    index = row_index(rows)
    write_plotdata(rows)
    plot_exp07(index)
    plot_multi(index, "Exp08", [f"ch_overload_factor={x}" for x in ("1.0", "1.5", "2.0", "3.0")],
               [1.0, 1.5, 2.0, 3.0], ("propagation_delay", "delivery_ratio"), "CH Overload Factor", "exp08_final_overload")
    plot_multi(index, "Exp09", [f"edge_prob={x}" for x in ("0.04", "0.06", "0.08", "0.10", "0.12")],
               [0.04, 0.06, 0.08, 0.10, 0.12], ("duplicates", "propagation_delay"), "ER Edge Probability (p)", "exp09_final_density")
    checked = verify_plotdata(rows)
    print("S6 input SHA-256: PASS")
    print("S6 cells validated: 42/42")
    print(f"plotted points checked: {checked}/{checked}")
    print("mean mismatches: 0")
    print("CI mismatches: 0")


if __name__ == "__main__":
    main()
