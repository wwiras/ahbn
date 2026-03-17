from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# Logging helpers
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_plot_start(name: str) -> None:
    print(f"\n[INFO] Plot ongoing: {name}", flush=True)


def log_plot_done(name: str) -> None:
    print(f"[INFO] Plot done: {name}", flush=True)


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------

def parse_result_dirname(results_dirname: str) -> str:
    if not results_dirname.startswith("results_"):
        raise ValueError("Result folder argument must start with 'results_'")
    return results_dirname.replace("results_", "", 1)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(results_dir: Path, name: str) -> pd.DataFrame:
    path = results_dir / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return pd.read_csv(path)


def save_plot(figures_dir: Path, filename: str) -> None:
    ensure_dir(figures_dir)
    plt.tight_layout()
    plt.savefig(figures_dir / filename, dpi=300, bbox_inches="tight")
    plt.close()


# -----------------------------------------------------------------------------
# Main plots
# -----------------------------------------------------------------------------

def plot_exp1_fanout_vs_duplication(results_dir: Path, figures_dir: Path) -> None:
    name = "Exp 1 - Fanout vs Duplication"
    log_plot_start(name)

    df = read_csv(results_dir, "exp1_fanout_vs_duplication.csv").sort_values("fanout")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_rounds"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 1: Fanout vs Propagation Delay")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp1_fanout_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_transmissions"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Transmissions")
    plt.title("Exp 1: Fanout vs Number of Messages")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp1_fanout_vs_transmissions.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_duplicates"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Duplicate Messages")
    plt.title("Exp 1: Fanout vs Duplicate Messages")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp1_fanout_vs_duplicates.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_duplicate_ratio"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Duplicate Ratio")
    plt.title("Exp 1: Duplicate Ratio vs Fanout")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp1_duplicate_ratio_vs_fanout.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["avg_duplicates"], df["avg_rounds"], marker="o")
    for _, row in df.iterrows():
        plt.annotate(f"k={int(row['fanout'])}", (row["avg_duplicates"], row["avg_rounds"]))
    plt.xlabel("Average Duplicate Messages")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 1: Propagation–Duplication Trade-off")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp1_delay_duplicate_tradeoff.png")

    log_plot_done(name)


def plot_exp2_ch_count_vs_node_count(results_dir: Path, figures_dir: Path) -> None:
    name = "Exp 2 - CH Count vs Node Count"
    log_plot_start(name)

    df = read_csv(results_dir, "exp2_ch_count_vs_node_count.csv").sort_values(["node_count", "ch_count"])

    for node_count in sorted(df["node_count"].unique()):
        sub = df[df["node_count"] == node_count].sort_values("ch_count")

        plt.figure(figsize=(8, 5))
        plt.plot(sub["ch_count"], sub["avg_rounds"], marker="o")
        plt.xlabel("Cluster Head Count")
        plt.ylabel("Average Propagation Delay (Rounds)")
        plt.title(f"Exp 2: CH Count vs Delay (Nodes={node_count})")
        plt.grid(True, alpha=0.3)
        save_plot(figures_dir, f"exp2_rounds_nodes_{node_count}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(sub["ch_count"], sub["avg_transmissions"], marker="o")
        plt.xlabel("Cluster Head Count")
        plt.ylabel("Average Transmissions")
        plt.title(f"Exp 2: CH Count vs Message Cost (Nodes={node_count})")
        plt.grid(True, alpha=0.3)
        save_plot(figures_dir, f"exp2_transmissions_nodes_{node_count}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(sub["ch_count"], sub["avg_delivery_ratio"], marker="o")
        plt.xlabel("Cluster Head Count")
        plt.ylabel("Average Delivery Ratio")
        plt.title(f"Exp 2: CH Count vs Coverage (Nodes={node_count})")
        plt.grid(True, alpha=0.3)
        save_plot(figures_dir, f"exp2_delivery_nodes_{node_count}.png")

    log_plot_done(name)


def plot_exp3_topology_density_vs_performance(results_dir: Path, figures_dir: Path) -> None:
    name = "Exp 3 - Topology Density vs Performance"
    log_plot_start(name)

    df = read_csv(results_dir, "exp3_topology_density_vs_performance.csv").sort_values("ba_m")

    plt.figure(figsize=(8, 5))
    plt.plot(df["ba_m"], df["avg_rounds"], marker="o")
    plt.xlabel("BA Parameter m")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 3: Topology Density vs Propagation Delay")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp3_density_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["ba_m"], df["avg_duplicates"], marker="o")
    plt.xlabel("BA Parameter m")
    plt.ylabel("Average Duplicate Messages")
    plt.title("Exp 3: Topology Density vs Duplicate Messages")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp3_density_vs_duplicates.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["ba_m"], df["avg_duplicate_ratio"], marker="o")
    plt.xlabel("BA Parameter m")
    plt.ylabel("Average Duplicate Ratio")
    plt.title("Exp 3: Topology Density vs Duplicate Ratio")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp3_density_vs_duplicate_ratio.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["avg_duplicates"], df["avg_rounds"], marker="o")
    for _, row in df.iterrows():
        plt.annotate(f"m={int(row['ba_m'])}", (row["avg_duplicates"], row["avg_rounds"]))
    plt.xlabel("Average Duplicate Messages")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 3: Density-Induced Trade-off")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp3_density_tradeoff.png")

    log_plot_done(name)


def plot_exp4_ch_overload_failure(results_dir: Path, figures_dir: Path) -> None:
    name = "Exp 4 - CH Overload / Failure"
    log_plot_start(name)

    df = read_csv(results_dir, "exp4_ch_overload_failure.csv")

    overload = df[df["scenario"] == "overload"].copy()
    failure = df[df["scenario"] == "failure"].copy()

    def overload_order(value):
        return 999 if str(value) == "full" else int(value)

    overload["sort_key"] = overload["overload_limit"].map(overload_order)
    overload = overload.sort_values("sort_key")

    plt.figure(figsize=(8, 5))
    plt.plot(overload["overload_limit"].astype(str), overload["avg_delivery_ratio"], marker="o")
    plt.xlabel("CH Service Limit")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 4: CH Overload vs Coverage")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp4_overload_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    plt.plot(overload["overload_limit"].astype(str), overload["avg_rounds"], marker="o")
    plt.xlabel("CH Service Limit")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 4: CH Overload vs Delay")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp4_overload_vs_rounds.png")

    failure = failure.sort_values("failed_chs")

    plt.figure(figsize=(8, 5))
    plt.plot(failure["failed_chs"], failure["avg_delivery_ratio"], marker="o")
    plt.xlabel("Failed CH Count")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 4: CH Failure vs Coverage")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp4_failure_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    plt.plot(failure["failed_chs"], failure["avg_rounds"], marker="o")
    plt.xlabel("Failed CH Count")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 4: CH Failure vs Delay")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp4_failure_vs_rounds.png")

    log_plot_done(name)


def plot_exp5_churn_sensitivity(results_dir: Path, figures_dir: Path) -> None:
    name = "Exp 5 - Churn Sensitivity"
    log_plot_start(name)

    df = read_csv(results_dir, "exp5_churn_sensitivity.csv").sort_values(["protocol", "churn_rate"])

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_delivery_ratio"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 5: Churn vs Coverage")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp5_churn_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_rounds"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 5: Churn vs Delay")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp5_churn_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_duplicates"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Duplicates")
    plt.title("Exp 5: Churn vs Duplicate Messages")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "exp5_churn_vs_duplicates.png")

    log_plot_done(name)


def plot_exp6_heterogeneous_resources(results_dir: Path, figures_dir: Path) -> None:
    name = "Exp 6 - Heterogeneous Resources"
    log_plot_start(name)

    df = read_csv(results_dir, "exp6_heterogeneous_resources.csv").sort_values("scenario")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_rounds"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Exp 6: Resource Heterogeneity vs Delay")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot(figures_dir, "exp6_heterogeneity_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_delivery_ratio"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 6: Resource Heterogeneity vs Coverage")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot(figures_dir, "exp6_heterogeneity_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_duplicates"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Duplicates")
    plt.title("Exp 6: Resource Heterogeneity vs Duplicate Messages")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot(figures_dir, "exp6_heterogeneity_vs_duplicates.png")

    log_plot_done(name)


def plot_master_summary(results_dir: Path, figures_dir: Path) -> None:
    name = "Master Summary"
    log_plot_start(name)

    exp1 = read_csv(results_dir, "exp1_fanout_vs_duplication.csv")
    exp3 = read_csv(results_dir, "exp3_topology_density_vs_performance.csv")
    exp5 = read_csv(results_dir, "exp5_churn_sensitivity.csv")
    exp6 = read_csv(results_dir, "exp6_heterogeneous_resources.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(exp1["avg_duplicates"], exp1["avg_rounds"], marker="o", label="Exp1 Fanout")
    plt.plot(exp3["avg_duplicates"], exp3["avg_rounds"], marker="o", label="Exp3 BA Density")

    gossip = exp5[exp5["protocol"] == "gossip"]
    cluster = exp5[exp5["protocol"] == "cluster"]
    plt.plot(gossip["avg_duplicates"], gossip["avg_rounds"], marker="o", label="Exp5 Gossip")
    plt.plot(cluster["avg_duplicates"], cluster["avg_rounds"], marker="o", label="Exp5 Cluster")

    plt.plot(exp6["avg_duplicates"], exp6["avg_rounds"], marker="o", label="Exp6 Resources")
    plt.xlabel("Average Duplicate Messages")
    plt.ylabel("Average Propagation Delay (Rounds)")
    plt.title("Overall Dissemination Trade-off Landscape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "master_tradeoff_landscape.png")

    log_plot_done(name)


# -----------------------------------------------------------------------------
# Extra analysis plots
# -----------------------------------------------------------------------------

def plot_exp1_extra_analysis(results_dir: Path, figures_dir: Path) -> None:
    name = "Extra Analysis - Exp 1"
    log_plot_start(name)

    df = read_csv(results_dir, "exp1_fanout_vs_duplication.csv").sort_values("fanout")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_propagation_efficiency"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Propagation Efficiency")
    plt.title("Extra Analysis: Fanout vs Propagation Efficiency")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "extra_exp1_fanout_vs_efficiency.png")

    log_plot_done(name)


def plot_exp3_extra_analysis(results_dir: Path, figures_dir: Path) -> None:
    name = "Extra Analysis - Exp 3"
    log_plot_start(name)

    df = read_csv(results_dir, "exp3_topology_density_vs_performance.csv").sort_values("ba_m")

    plt.figure(figsize=(8, 5))
    plt.plot(df["ba_m"], df["avg_path_length"], marker="o")
    plt.xlabel("BA Parameter m")
    plt.ylabel("Average Path Length")
    plt.title("Extra Analysis: Topology Density vs Average Path Length")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "extra_exp3_density_vs_path_length.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["ba_m"], df["avg_degree"], marker="o")
    plt.xlabel("BA Parameter m")
    plt.ylabel("Average Degree")
    plt.title("Extra Analysis: Topology Density vs Average Degree")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "extra_exp3_density_vs_avg_degree.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["ba_m"], df["avg_clustering_coeff"], marker="o")
    plt.xlabel("BA Parameter m")
    plt.ylabel("Average Clustering Coefficient")
    plt.title("Extra Analysis: Topology Density vs Clustering Coefficient")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "extra_exp3_density_vs_clustering_coeff.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["avg_clustering_coeff"], df["avg_duplicates"], marker="o")
    for _, row in df.iterrows():
        plt.annotate(f"m={int(row['ba_m'])}", (row["avg_clustering_coeff"], row["avg_duplicates"]))
    plt.xlabel("Average Clustering Coefficient")
    plt.ylabel("Average Duplicates")
    plt.title("Extra Analysis: Clustering Coefficient vs Duplicates")
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "extra_exp3_clustering_vs_duplicates.png")

    log_plot_done(name)


def plot_exp5_extra_analysis(results_dir: Path, figures_dir: Path) -> None:
    name = "Extra Analysis - Exp 5"
    log_plot_start(name)

    df = read_csv(results_dir, "exp5_churn_sensitivity.csv").sort_values(["protocol", "churn_rate"])

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_propagation_efficiency"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Propagation Efficiency")
    plt.title("Extra Analysis: Churn vs Propagation Efficiency")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot(figures_dir, "extra_exp5_churn_vs_efficiency.png")

    log_plot_done(name)


def plot_exp6_extra_analysis(results_dir: Path, figures_dir: Path) -> None:
    name = "Extra Analysis - Exp 6"
    log_plot_start(name)

    df = read_csv(results_dir, "exp6_heterogeneous_resources.csv").sort_values("scenario")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_propagation_efficiency"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Propagation Efficiency")
    plt.title("Extra Analysis: Resource Heterogeneity vs Propagation Efficiency")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot(figures_dir, "extra_exp6_heterogeneity_vs_efficiency.png")

    log_plot_done(name)


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    if len(sys.argv) != 2:
        print("Usage: python plot_result.py <results_folder_name>")
        print("Example: python plot_result.py results_2026Mar17_0804")
        sys.exit(1)

    results_dirname = sys.argv[1]
    results_dir = Path(results_dirname)

    if not results_dir.exists() or not results_dir.is_dir():
        print(f"[ERROR] Results folder does not exist: {results_dir.resolve()}", flush=True)
        sys.exit(1)

    suffix = parse_result_dirname(results_dirname)
    figures_dir = Path(f"figures_{suffix}")
    ensure_dir(figures_dir)

    log(f"Reading results from: {results_dir.resolve()}")
    log(f"Figures folder ready: {figures_dir.resolve()}")

    plot_exp1_fanout_vs_duplication(results_dir, figures_dir)
    plot_exp2_ch_count_vs_node_count(results_dir, figures_dir)
    plot_exp3_topology_density_vs_performance(results_dir, figures_dir)
    plot_exp4_ch_overload_failure(results_dir, figures_dir)
    plot_exp5_churn_sensitivity(results_dir, figures_dir)
    plot_exp6_heterogeneous_resources(results_dir, figures_dir)
    plot_master_summary(results_dir, figures_dir)

    plot_exp1_extra_analysis(results_dir, figures_dir)
    plot_exp3_extra_analysis(results_dir, figures_dir)
    plot_exp5_extra_analysis(results_dir, figures_dir)
    plot_exp6_extra_analysis(results_dir, figures_dir)

    log("Plotting pipeline completed successfully")


if __name__ == "__main__":
    main()