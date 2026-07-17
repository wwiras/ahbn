from __future__ import annotations

from pathlib import Path
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RESULTS_DIR = Path("results")
FIGURES_DIR = Path("figures")

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

def read_csv(name: str) -> pd.DataFrame:
    path = RESULTS_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}. Run ahbn_experiments_1_to_6_ba.py first.")
    return pd.read_csv(path)

def save_plot(fig, filename: str) -> None:
    """Saves the plotly figure as an interactive HTML file."""
    ensure_dir(FIGURES_DIR)
    # You can also use fig.write_image(str(FIGURES_DIR / filename)) if you have kaleido installed
    html_path = FIGURES_DIR / filename.replace(".png", ".html")
    fig.write_html(str(html_path))

# --- Experiment Plotting Functions ---

def plot_exp1_fanout_vs_duplication() -> None:
    df = read_csv("exp1_fanout_vs_duplication.csv").sort_values("fanout")

    # Example 1: Rounds
    fig1 = px.line(df, x="fanout", y="avg_rounds", markers=True, 
                   title="Exp 1: Fanout vs Propagation Delay",
                   labels={"fanout": "Fanout", "avg_rounds": "Avg Rounds"})
    save_plot(fig1, "exp1_fanout_vs_rounds.png")

    # Example 2: Transmissions
    fig2 = px.line(df, x="fanout", y="avg_transmissions", markers=True, 
                   title="Exp 1: Fanout vs Number of Messages")
    save_plot(fig2, "exp1_fanout_vs_transmissions.png")

    # Trade-off Plot with dynamic labels (Intuitive Hover)
    fig3 = px.line(df, x="avg_duplicates", y="avg_rounds", markers=True,
                   hover_data=["fanout"],
                   title="Exp 1: Propagation–Duplication Trade-off",
                   labels={"avg_duplicates": "Avg Duplicate Messages", "avg_rounds": "Avg Rounds"})
    save_plot(fig3, "exp1_delay_duplicate_tradeoff.png")

def plot_exp2_ch_count_vs_node_count() -> None:
    df = read_csv("exp2_ch_count_vs_node_count.csv").sort_values(["node_count", "ch_count"])

    for node_count in sorted(df["node_count"].unique()):
        sub = df[df["node_count"] == node_count].sort_values("ch_count")
        
        fig = px.line(sub, x="ch_count", y="avg_rounds", markers=True,
                      title=f"Exp 2: CH Count vs Delay (Nodes={node_count})")
        save_plot(fig, f"exp2_rounds_nodes_{node_count}.png")

def plot_exp3_topology_density_vs_performance() -> None:
    df = read_csv("exp3_topology_density_vs_performance.csv").sort_values("ba_m")
    
    fig = px.line(df, x="ba_m", y="avg_rounds", markers=True,
                  title="Exp 3: Topology Density vs Propagation Delay")
    save_plot(fig, "exp3_density_vs_rounds.png")

def plot_exp4_ch_overload_failure() -> None:
    df = read_csv("exp4_ch_overload_failure.csv")
    overload = df[df["scenario"] == "overload"].copy()
    failure = df[df["scenario"] == "failure"].copy()

    # Handle custom sorting for "full" capacity
    overload["sort_key"] = overload["overload_limit"].map(lambda x: 999 if str(x) == "full" else int(x))
    overload = overload.sort_values("sort_key")

    fig_overload = px.line(overload, x="overload_limit", y="avg_delivery_ratio", markers=True,
                           title="Exp 4: CH Overload vs Coverage")
    save_plot(fig_overload, "exp4_overload_vs_delivery.png")

    failure = failure.sort_values("failed_chs")
    fig_fail = px.line(failure, x="failed_chs", y="avg_delivery_ratio", markers=True,
                        title="Exp 4: CH Failure vs Coverage")
    save_plot(fig_fail, "exp4_failure_vs_delivery.png")

def plot_exp5_churn_sensitivity() -> None:
    df = read_csv("exp5_churn_sensitivity.csv").sort_values(["protocol", "churn_rate"])

    # Plotly handles multi-line "for loops" automatically via the 'color' argument
    fig = px.line(df, x="churn_rate", y="avg_delivery_ratio", color="protocol", markers=True,
                  title="Exp 5: Churn vs Coverage")
    save_plot(fig, "exp5_churn_vs_delivery.png")

def plot_exp6_heterogeneous_resources() -> None:
    df = read_csv("exp6_heterogeneous_resources.csv").sort_values("scenario")

    # Simple Bar Chart
    fig = px.bar(df, x="scenario", y="avg_rounds", color="scenario",
                 title="Exp 6: Resource Heterogeneity vs Delay")
    save_plot(fig, "exp6_heterogeneity_vs_rounds.png")

def plot_master_summary() -> None:
    # Combining multiple dataframes into one for Plotly's long-form preference
    exp1 = read_csv("exp1_fanout_vs_duplication.csv"); exp1["Source"] = "Exp1 Fanout"
    exp3 = read_csv("exp3_topology_density_vs_performance.csv"); exp3["Source"] = "Exp3 Density"
    
    # Using Graph Objects for complex manual overlay if preferred, 
    # but we can just use a simple concat for px
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=exp1["avg_duplicates"], y=exp1["avg_rounds"], name="Exp1 Fanout", mode='lines+markers'))
    fig.add_trace(go.Scatter(x=exp3["avg_duplicates"], y=exp3["avg_rounds"], name="Exp3 BA Density", mode='lines+markers'))
    
    fig.update_layout(title="Overall Dissemination Trade-off Landscape",
                      xaxis_title="Avg Duplicate Messages",
                      yaxis_title="Avg Propagation Rounds",
                      template="plotly_white")
    
    save_plot(fig, "master_tradeoff_landscape.png")

def main() -> None:
    ensure_dir(FIGURES_DIR)
    
    # Execute all
    plot_exp1_fanout_vs_duplication()
    plot_exp2_ch_count_vs_node_count()
    plot_exp3_topology_density_vs_performance()
    plot_exp4_ch_overload_failure()
    plot_exp5_churn_sensitivity()
    plot_exp6_heterogeneous_resources()
    plot_master_summary()

    print(f"Interactive HTML figures written to: {FIGURES_DIR.resolve()}")

if __name__ == "__main__":
    main()