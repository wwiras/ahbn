from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run ahbn_experiments_1_to_6.py first.")
    return pd.read_csv(path)


def save_plot(filename: str) -> None:
    ensure_dir(FIGURES_DIR)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / filename, dpi=300, bbox_inches="tight")
    plt.close()


def plot_exp1_fanout_vs_duplication() -> None:
    df = read_csv("exp1_fanout_vs_duplication.csv").sort_values("fanout")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_duplicates"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Duplicates")
    plt.title("Exp 1: Fanout vs Duplication")
    plt.grid(True, alpha=0.3)
    save_plot("exp1_fanout_vs_duplication.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["fanout"], df["avg_rounds"], marker="o")
    plt.xlabel("Fanout")
    plt.ylabel("Average Rounds")
    plt.title("Exp 1: Fanout vs Propagation Rounds")
    plt.grid(True, alpha=0.3)
    save_plot("exp1_fanout_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["avg_duplicates"], df["avg_rounds"], marker="o")
    for _, row in df.iterrows():
        plt.annotate(f"k={int(row['fanout'])}", (row["avg_duplicates"], row["avg_rounds"]))
    plt.xlabel("Average Duplicates")
    plt.ylabel("Average Rounds")
    plt.title("Exp 1: Delay vs Duplicate Trade-off")
    plt.grid(True, alpha=0.3)
    save_plot("exp1_delay_duplicate_tradeoff.png")


def plot_exp2_ch_count_vs_node_count() -> None:
    df = read_csv("exp2_ch_count_vs_node_count.csv").sort_values(["node_count", "ch_count"])

    for node_count in sorted(df["node_count"].unique()):
        sub = df[df["node_count"] == node_count].sort_values("ch_count")

        plt.figure(figsize=(8, 5))
        plt.plot(sub["ch_count"], sub["avg_rounds"], marker="o")
        plt.xlabel("Cluster Head Count")
        plt.ylabel("Average Rounds")
        plt.title(f"Exp 2: CH Count vs Rounds (Nodes={node_count})")
        plt.grid(True, alpha=0.3)
        save_plot(f"exp2_rounds_nodes_{node_count}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(sub["ch_count"], sub["avg_transmissions"], marker="o")
        plt.xlabel("Cluster Head Count")
        plt.ylabel("Average Transmissions")
        plt.title(f"Exp 2: CH Count vs Transmissions (Nodes={node_count})")
        plt.grid(True, alpha=0.3)
        save_plot(f"exp2_transmissions_nodes_{node_count}.png")

        plt.figure(figsize=(8, 5))
        plt.plot(sub["ch_count"], sub["avg_delivery_ratio"], marker="o")
        plt.xlabel("Cluster Head Count")
        plt.ylabel("Average Delivery Ratio")
        plt.title(f"Exp 2: CH Count vs Delivery Ratio (Nodes={node_count})")
        plt.grid(True, alpha=0.3)
        save_plot(f"exp2_delivery_nodes_{node_count}.png")

    plt.figure(figsize=(8, 5))
    for node_count in sorted(df["node_count"].unique()):
        sub = df[df["node_count"] == node_count].sort_values("ch_count")
        plt.plot(sub["ch_count"], sub["avg_rounds"], marker="o", label=f"N={node_count}")
    plt.xlabel("Cluster Head Count")
    plt.ylabel("Average Rounds")
    plt.title("Exp 2: CH Count vs Rounds Across Node Counts")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("exp2_rounds_all_nodes.png")


def plot_exp3_topology_density_vs_performance() -> None:
    df = read_csv("exp3_topology_density_vs_performance.csv").sort_values("degree_hint")

    plt.figure(figsize=(8, 5))
    plt.plot(df["degree_hint"], df["avg_rounds"], marker="o")
    plt.xlabel("Topology Density (Degree Hint)")
    plt.ylabel("Average Rounds")
    plt.title("Exp 3: Topology Density vs Propagation Rounds")
    plt.grid(True, alpha=0.3)
    save_plot("exp3_density_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["degree_hint"], df["avg_duplicates"], marker="o")
    plt.xlabel("Topology Density (Degree Hint)")
    plt.ylabel("Average Duplicates")
    plt.title("Exp 3: Topology Density vs Duplicates")
    plt.grid(True, alpha=0.3)
    save_plot("exp3_density_vs_duplicates.png")

    plt.figure(figsize=(8, 5))
    plt.plot(df["avg_duplicates"], df["avg_rounds"], marker="o")
    for _, row in df.iterrows():
        plt.annotate(f"d={int(row['degree_hint'])}", (row["avg_duplicates"], row["avg_rounds"]))
    plt.xlabel("Average Duplicates")
    plt.ylabel("Average Rounds")
    plt.title("Exp 3: Density-Induced Trade-off")
    plt.grid(True, alpha=0.3)
    save_plot("exp3_density_tradeoff.png")


def plot_exp4_ch_overload_failure() -> None:
    df = read_csv("exp4_ch_overload_failure.csv")

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
    plt.title("Exp 4: CH Overload vs Delivery Ratio")
    plt.grid(True, alpha=0.3)
    save_plot("exp4_overload_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    plt.plot(overload["overload_limit"].astype(str), overload["avg_transmissions"], marker="o")
    plt.xlabel("CH Service Limit")
    plt.ylabel("Average Transmissions")
    plt.title("Exp 4: CH Overload vs Transmissions")
    plt.grid(True, alpha=0.3)
    save_plot("exp4_overload_vs_transmissions.png")

    failure = failure.sort_values("failed_chs")

    plt.figure(figsize=(8, 5))
    plt.plot(failure["failed_chs"], failure["avg_delivery_ratio"], marker="o")
    plt.xlabel("Failed CH Count")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 4: CH Failure vs Delivery Ratio")
    plt.grid(True, alpha=0.3)
    save_plot("exp4_failure_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    plt.plot(failure["failed_chs"], failure["avg_rounds"], marker="o")
    plt.xlabel("Failed CH Count")
    plt.ylabel("Average Rounds")
    plt.title("Exp 4: CH Failure vs Rounds")
    plt.grid(True, alpha=0.3)
    save_plot("exp4_failure_vs_rounds.png")


def plot_exp5_churn_sensitivity() -> None:
    df = read_csv("exp5_churn_sensitivity.csv").sort_values(["protocol", "churn_rate"])

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_delivery_ratio"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 5: Churn vs Delivery Ratio")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("exp5_churn_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_rounds"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Rounds")
    plt.title("Exp 5: Churn vs Propagation Rounds")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("exp5_churn_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    for protocol in sorted(df["protocol"].unique()):
        sub = df[df["protocol"] == protocol].sort_values("churn_rate")
        plt.plot(sub["churn_rate"], sub["avg_duplicates"], marker="o", label=protocol)
    plt.xlabel("Churn Rate")
    plt.ylabel("Average Duplicates")
    plt.title("Exp 5: Churn vs Duplicates")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("exp5_churn_vs_duplicates.png")


def plot_exp6_heterogeneous_resources() -> None:
    df = read_csv("exp6_heterogeneous_resources.csv").sort_values("scenario")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_rounds"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Rounds")
    plt.title("Exp 6: Resource Heterogeneity vs Rounds")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot("exp6_heterogeneity_vs_rounds.png")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_delivery_ratio"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Delivery Ratio")
    plt.title("Exp 6: Resource Heterogeneity vs Delivery Ratio")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot("exp6_heterogeneity_vs_delivery.png")

    plt.figure(figsize=(8, 5))
    plt.bar(df["scenario"], df["avg_duplicates"])
    plt.xlabel("Scenario")
    plt.ylabel("Average Duplicates")
    plt.title("Exp 6: Resource Heterogeneity vs Duplicates")
    plt.grid(True, alpha=0.3, axis="y")
    save_plot("exp6_heterogeneity_vs_duplicates.png")


def plot_master_summary() -> None:
    exp1 = read_csv("exp1_fanout_vs_duplication.csv")
    exp3 = read_csv("exp3_topology_density_vs_performance.csv")
    exp5 = read_csv("exp5_churn_sensitivity.csv")
    exp6 = read_csv("exp6_heterogeneous_resources.csv")

    plt.figure(figsize=(8, 5))
    plt.plot(exp1["avg_duplicates"], exp1["avg_rounds"], marker="o", label="Exp1 Fanout")
    plt.plot(exp3["avg_duplicates"], exp3["avg_rounds"], marker="o", label="Exp3 Density")

    gossip = exp5[exp5["protocol"] == "gossip"]
    cluster = exp5[exp5["protocol"] == "cluster"]
    plt.plot(gossip["avg_duplicates"], gossip["avg_rounds"], marker="o", label="Exp5 Gossip")
    plt.plot(cluster["avg_duplicates"], cluster["avg_rounds"], marker="o", label="Exp5 Cluster")

    plt.plot(exp6["avg_duplicates"], exp6["avg_rounds"], marker="o", label="Exp6 Resources")
    plt.xlabel("Average Duplicates")
    plt.ylabel("Average Rounds")
    plt.title("Overall Dissemination Trade-off Landscape")
    plt.legend()
    plt.grid(True, alpha=0.3)
    save_plot("master_tradeoff_landscape.png")


def main() -> None:
    ensure_dir(FIGURES_DIR)

    plot_exp1_fanout_vs_duplication()
    plot_exp2_ch_count_vs_node_count()
    plot_exp3_topology_density_vs_performance()
    plot_exp4_ch_overload_failure()
    plot_exp5_churn_sensitivity()
    plot_exp6_heterogeneous_resources()
    plot_master_summary()

    print(f"Figures written to: {FIGURES_DIR.resolve()}")


if __name__ == "__main__":
    main()