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

    # support both old single-protocol and new multi-protocol style
    if "protocols" in config:
        protocols = config["protocols"]
    else:
        protocols = [config["protocol"]]

    runs = config["runs_per_setting"]

    node_counts = config.get("node_counts", [None])
    fanouts = config.get("fanouts", [None])
    ch_counts = config.get("ch_counts", [None])
    ba_ms = config.get("ba_m_values", [2])
    scenarios = config.get("scenarios", ["normal"])

    rows = []

    # IMPORTANT:
    # seed loop is outside protocol loop
    # so each protocol gets the same seed under the same parameter setting
    grid = itertools.product(node_counts, fanouts, ch_counts, ba_ms, scenarios)

    for node_count, fanout, ch_count, ba_m, scenario in grid:
        for seed in range(1, runs + 1):
            for protocol in protocols:

                result = run_simulation(
                    protocol=protocol,
                    node_count=node_count,
                    fanout=fanout,
                    ch_count=ch_count,
                    ba_m=ba_m,
                    scenario=scenario,
                    seed=seed
                )

                rows.append(result)

    df = pd.DataFrame(rows)

    os.makedirs("results/raw", exist_ok=True)
    os.makedirs("results/summary", exist_ok=True)
    os.makedirs("results/figures", exist_ok=True)

    raw_file = f"results/raw/{experiment_id}.csv"
    df.to_csv(raw_file, index=False)

    summary = make_summary(df)
    summary_file = f"results/summary/{experiment_id}_summary.csv"
    summary.to_csv(summary_file, index=False)

    generate_figures(experiment_id, df)

    print("Experiment finished:", experiment_id)
    print("Raw:", raw_file)
    print("Summary:", summary_file)


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python run_one.py configs/exp07.yaml")
        exit()

    run_experiment(sys.argv[1])