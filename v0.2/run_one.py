import yaml
import pandas as pd
import itertools
import os
import sys

from simulator import run_simulation
from analysis import make_summary, generate_figures

def run_experiment(config_file):

    with open(config_file) as f:
        config = yaml.safe_load(f)

    experiment_id = config["experiment_id"]
    protocol = config["protocol"]
    runs = config["runs_per_setting"]

    node_counts = config.get("node_counts",[None])
    fanouts = config.get("fanouts",[None])
    ch_counts = config.get("ch_counts",[None])
    ba_ms = config.get("ba_m_values",[2])
    scenarios = config.get("scenarios",["normal"])

    rows = []

    grid = itertools.product(node_counts, fanouts, ch_counts, ba_ms, scenarios)

    for node_count, fanout, ch_count, ba_m, scenario in grid:

        for seed in range(1, runs + 1):

            result = run_simulation(
                protocol,
                node_count,
                fanout,
                ch_count,
                ba_m,
                scenario,
                seed
            )

            rows.append(result)

    df = pd.DataFrame(rows)

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)

    raw_file = f"results/raw/{experiment_id}.csv"
    df.to_csv(raw_file, index=False)

    summary = make_summary(df)
    summary_file = f"results/summary/{experiment_id}_summary.csv"
    summary.to_csv(summary_file, index=False)

    generate_figures(experiment_id, df)

    print("Experiment finished:", experiment_id)

if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python run_one.py configs/exp01.yaml")
        exit()

    run_experiment(sys.argv[1])