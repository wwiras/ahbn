from __future__ import annotations

import argparse
import copy
import sys
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import yaml


# Allow execution from the repository root as:
#   python scripts/run_stage2_sensitivity.py --config configs/stage2_parameter_sensitivity.yaml
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from run_batch import run_single  # noqa: E402


PRIMARY_PARAMETERS = ("alpha", "kappa", "beta", "mode_threshold")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def build_run_cfg(
    stage2_cfg: dict[str, Any],
    scenario_name: str,
    parameter_name: str,
    parameter_value: float,
) -> dict[str, Any]:
    """Build one isolated run configuration without mutating the source YAML."""
    scenario = copy.deepcopy(stage2_cfg["scenarios"][scenario_name])
    canonical_ahbn = copy.deepcopy(stage2_cfg["ahbn"])

    if parameter_name not in canonical_ahbn:
        raise KeyError(
            f"Sensitivity parameter '{parameter_name}' is not present in the canonical ahbn block."
        )

    canonical_ahbn[parameter_name] = parameter_value

    # run_single() only special-cases Exp12 resource assignment.  A Stage-2
    # experiment label is therefore safe and keeps trace rows clearly tagged.
    run_cfg: dict[str, Any] = {
        "experiment": "stage2_parameter_sensitivity",
        "ahbn": canonical_ahbn,
        "failure": copy.deepcopy(scenario.get("failure", {"enabled": False})),
        "churn": copy.deepcopy(scenario.get("churn", {"enabled": False})),
    }

    return run_cfg


def summarize_trace(trace_rows: list[Any]) -> dict[str, Any]:
    """Reduce an adaptive trace to compact per-run controller diagnostics."""
    if not trace_rows:
        return {
            "trace_row_count": 0,
            "mean_score": None,
            "mean_weight": None,
            "gossip_fraction": None,
            "cluster_fraction": None,
            "mean_fanout": None,
            "min_observed_fanout": None,
            "max_observed_fanout": None,
        }

    df = pd.DataFrame([asdict(row) for row in trace_rows])
    total = len(df)

    return {
        "trace_row_count": total,
        "mean_score": float(df["score"].mean()),
        "mean_weight": float(df["weight"].mean()),
        "gossip_fraction": float((df["mode"] == "gossip").sum() / total),
        "cluster_fraction": float((df["mode"] == "cluster").sum() / total),
        "mean_fanout": float(df["fanout"].mean()),
        "min_observed_fanout": int(df["fanout"].min()),
        "max_observed_fanout": int(df["fanout"].max()),
    }


def validate_config(cfg: dict[str, Any]) -> None:
    required_top = {"seed", "runs_per_setting", "ahbn", "sensitivity", "scenarios"}
    missing = required_top - set(cfg)
    if missing:
        raise ValueError(f"Stage-2 config is missing: {sorted(missing)}")

    for parameter in PRIMARY_PARAMETERS:
        if parameter not in cfg["sensitivity"]:
            raise ValueError(f"Missing primary sensitivity sweep: {parameter}")
        if parameter not in cfg["ahbn"]:
            raise ValueError(f"Missing canonical AHBN value: {parameter}")

    for scenario_name, scenario in cfg["scenarios"].items():
        for key in ("topology_type", "num_nodes"):
            if key not in scenario:
                raise ValueError(f"Scenario '{scenario_name}' is missing '{key}'.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Stage 2 AHBN one-factor-at-a-time parameter sensitivity."
    )
    parser.add_argument(
        "--config",
        default="configs/stage2_parameter_sensitivity.yaml",
        help="Stage-2 YAML configuration file.",
    )
    parser.add_argument(
        "--parameter",
        action="append",
        choices=PRIMARY_PARAMETERS,
        help="Run only this parameter. May be supplied more than once.",
    )
    parser.add_argument(
        "--scenario",
        action="append",
        help="Run only this scenario. May be supplied more than once.",
    )
    parser.add_argument(
        "--runs-per-setting",
        type=int,
        help="Override YAML runs_per_setting (useful for a 1-run smoke test).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_yaml(args.config)
    validate_config(cfg)

    parameters = args.parameter or list(PRIMARY_PARAMETERS)
    scenarios = args.scenario or list(cfg["scenarios"].keys())

    unknown_scenarios = [name for name in scenarios if name not in cfg["scenarios"]]
    if unknown_scenarios:
        raise ValueError(f"Unknown scenario(s): {unknown_scenarios}")

    runs_per_setting = (
        args.runs_per_setting
        if args.runs_per_setting is not None
        else int(cfg["runs_per_setting"])
    )
    if runs_per_setting < 1:
        raise ValueError("runs_per_setting must be >= 1")

    base_seed = int(cfg["seed"])
    enable_trace = bool(cfg.get("enable_adaptive_trace", True))
    save_raw_trace = bool(cfg.get("save_adaptive_trace", False))

    result_rows: list[dict[str, Any]] = []
    raw_trace_rows: list[dict[str, Any]] = []

    total_settings = sum(len(cfg["sensitivity"][p]) for p in parameters) * len(scenarios)
    total_runs = total_settings * runs_per_setting

    print("Stage 2 — AHBN Parameter Sensitivity")
    print(f"Parameters : {', '.join(parameters)}")
    print(f"Scenarios  : {', '.join(scenarios)}")
    print(f"Runs/setting: {runs_per_setting}")
    print(f"Total AHBN runs: {total_runs}")
    print()

    completed = 0

    for scenario_name in scenarios:
        scenario = cfg["scenarios"][scenario_name]

        for parameter_name in parameters:
            canonical_value = cfg["ahbn"][parameter_name]
            values = cfg["sensitivity"][parameter_name]

            for parameter_value in values:
                for run_idx in range(runs_per_setting):
                    seed = base_seed + run_idx
                    run_cfg = build_run_cfg(
                        cfg,
                        scenario_name=scenario_name,
                        parameter_name=parameter_name,
                        parameter_value=parameter_value,
                    )

                    topology_type = scenario["topology_type"]
                    edge_prob = scenario.get("edge_prob")
                    ba_m = scenario.get("ba_m")
                    fanout = scenario.get("fanout")
                    num_clusters = scenario.get("num_clusters", 4)
                    churn_rate = scenario.get("churn_rate")
                    overload = scenario.get("ch_overload_factor", 1.0)

                    scenario_tag = (
                        f"scenario={scenario_name};"
                        f"parameter={parameter_name};"
                        f"value={parameter_value}"
                    )

                    summary = run_single(
                        cfg=run_cfg,
                        strategy_name="ahbn",
                        seed=seed,
                        topology_type=topology_type,
                        num_nodes=int(scenario["num_nodes"]),
                        use_topology_cache=bool(scenario.get("use_topology_cache", True)),
                        base_delay=float(scenario.get("base_delay", 1.0)),
                        jitter=float(scenario.get("jitter", 0.2)),
                        message_source=int(scenario.get("message_source", 0)),
                        fanout=fanout,
                        num_clusters=num_clusters,
                        ch_overload_factor=overload,
                        edge_prob=edge_prob,
                        ba_m=ba_m,
                        enable_adaptive_trace=enable_trace,
                        churn_rate=churn_rate,
                        scenario_tag=scenario_tag,
                    )

                    trace_rows = summary.pop("adaptive_trace_rows", [])
                    trace_summary = summarize_trace(trace_rows)

                    result_rows.append(
                        {
                            "experiment": "stage2_parameter_sensitivity",
                            "scenario": scenario_name,
                            "source_experiment": scenario.get("source_experiment"),
                            "scenario_description": scenario.get("description"),
                            "parameter": parameter_name,
                            "parameter_value": parameter_value,
                            "canonical_value": canonical_value,
                            "is_canonical_value": parameter_value == canonical_value,
                            "seed": seed,
                            "run_index": run_idx,
                            "num_nodes": int(scenario["num_nodes"]),
                            "topology_type": topology_type,
                            "edge_prob": edge_prob,
                            "ba_m": ba_m,
                            "configured_fanout": fanout,
                            "num_clusters": num_clusters,
                            "ch_overload_factor": overload,
                            "churn_rate": churn_rate,
                            "delivery_ratio": summary["delivery_ratio"],
                            "propagation_delay": summary["propagation_delay"],
                            "duplicates": summary["duplicates"],
                            "total_forwards": summary["total_forwards"],
                            "mode_switch_count": summary["mode_switch_count"],
                            "fanout_change_count": summary["fanout_change_count"],
                            "adaptation_event_count": summary["adaptation_event_count"],
                            **trace_summary,
                        }
                    )

                    if save_raw_trace and trace_rows:
                        for row in trace_rows:
                            trace_dict = asdict(row)
                            trace_dict.update(
                                {
                                    "sensitivity_scenario": scenario_name,
                                    "sensitivity_parameter": parameter_name,
                                    "sensitivity_value": parameter_value,
                                    "canonical_value": canonical_value,
                                }
                            )
                            raw_trace_rows.append(trace_dict)

                    completed += 1
                    print(
                        f"[{completed:>4}/{total_runs}] "
                        f"{scenario_name:<10} "
                        f"{parameter_name:<14}={str(parameter_value):<4} "
                        f"seed={seed} "
                        f"delivery={summary['delivery_ratio']:.3f} "
                        f"delay={summary['propagation_delay']} "
                        f"dup={summary['duplicates']}"
                    )

    output_dir = ROOT / "outputs" / "csv"
    output_dir.mkdir(parents=True, exist_ok=True)
    ts = timestamp()

    results_path = output_dir / f"stage2_parameter_sensitivity_{ts}.csv"
    pd.DataFrame(result_rows).to_csv(results_path, index=False)
    print(f"\nSaved results: {results_path}")

    if save_raw_trace and raw_trace_rows:
        trace_path = output_dir / f"stage2_parameter_sensitivity_trace_{ts}.csv"
        pd.DataFrame(raw_trace_rows).to_csv(trace_path, index=False)
        print(f"Saved trace  : {trace_path}")


if __name__ == "__main__":
    main()