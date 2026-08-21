from __future__ import annotations

import copy
import sys
from dataclasses import asdict
from pathlib import Path

import pandas as pd
import yaml
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ahbn.strategies.ahbn import AHBNStrategy  # noqa: E402
from run_batch import run_single  # noqa: E402


CONFIG_PATH = ROOT / "configs" / "stage2_parameter_sensitivity.yaml"
RAW_PATH = ROOT / "outputs" / "csv" / "fanout6_validation_raw.csv"
TRACE_PATH = ROOT / "outputs" / "csv" / "fanout6_validation_trace.csv"
SUMMARY_PATH = ROOT / "outputs" / "csv" / "fanout6_validation_summary.csv"
SCENARIOS = ("dense", "bottleneck", "churn")
MAX_FANOUTS = (4, 6)
METRICS = ("delivery_ratio", "propagation_delay", "duplicates", "total_forwards")


forwarding_diagnostics: dict[str, int] = {}
original_select_targets = AHBNStrategy.select_targets


def observed_select_targets(self, node, message, simulator):
    targets = original_select_targets(self, node, message, simulator)
    requested = self._get_effective_fanout(node)
    active_physical_degree = sum(
        neighbor_id in simulator.nodes and simulator.nodes[neighbor_id].is_active
        for neighbor_id in node.neighbors
        if neighbor_id != node.node_id
    )
    forwarding_diagnostics["decisions"] += 1
    forwarding_diagnostics["max_requested"] = max(
        forwarding_diagnostics["max_requested"], requested
    )
    forwarding_diagnostics["max_effective"] = max(
        forwarding_diagnostics["max_effective"], len(targets)
    )
    forwarding_diagnostics["max_physical_degree_excess"] = max(
        forwarding_diagnostics["max_physical_degree_excess"],
        len(targets) - active_physical_degree,
    )
    forwarding_diagnostics["max_budget_excess"] = max(
        forwarding_diagnostics["max_budget_excess"], len(targets) - requested
    )
    return targets


def mean_ci(series: pd.Series) -> tuple[float, float]:
    values = pd.to_numeric(series, errors="coerce").dropna()
    mean = float(values.mean())
    if len(values) < 2:
        return mean, 0.0
    sem = float(values.sem(ddof=1))
    return mean, float(t.ppf(0.975, len(values) - 1) * sem)


def decreased_after_above_four(trace: pd.DataFrame) -> bool:
    for _, node_rows in trace.sort_values(["node_id", "time"]).groupby("node_id"):
        values = node_rows["fanout"].astype(int).tolist()
        for index, value in enumerate(values):
            if value > 4 and any(later < value for later in values[index + 1 :]):
                return True
    return False


def main() -> None:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        stage2 = yaml.safe_load(handle)

    assert tuple(stage2["scenarios"].keys()) == SCENARIOS
    assert int(stage2["seed"]) == 42
    assert int(stage2["runs_per_setting"]) == 20
    assert int(stage2["ahbn"]["min_fanout"]) == 2
    assert int(stage2["ahbn"]["max_fanout"]) == 4
    assert int(stage2["scenarios"]["bottleneck"]["ba_m"]) == 3

    RAW_PATH.parent.mkdir(parents=True, exist_ok=True)
    raw_rows: list[dict] = []
    trace_rows: list[dict] = []
    expected_runs = len(SCENARIOS) * len(MAX_FANOUTS) * int(stage2["runs_per_setting"])
    completed = 0

    AHBNStrategy.select_targets = observed_select_targets
    try:
        for scenario_name in SCENARIOS:
            scenario = stage2["scenarios"][scenario_name]
            for max_fanout in MAX_FANOUTS:
                for run_index in range(int(stage2["runs_per_setting"])):
                    seed = int(stage2["seed"]) + run_index
                    run_cfg = {
                        "experiment": "fanout6_amendment_validation",
                        "ahbn": copy.deepcopy(stage2["ahbn"]),
                        "failure": copy.deepcopy(scenario["failure"]),
                        "churn": copy.deepcopy(scenario["churn"]),
                    }
                    run_cfg["ahbn"]["max_fanout"] = max_fanout
                    forwarding_diagnostics.clear()
                    forwarding_diagnostics.update(
                        decisions=0,
                        max_requested=0,
                        max_effective=0,
                        max_physical_degree_excess=0,
                        max_budget_excess=0,
                    )
                    summary = run_single(
                        cfg=run_cfg,
                        strategy_name="ahbn",
                        seed=seed,
                        topology_type=scenario["topology_type"],
                        num_nodes=int(scenario["num_nodes"]),
                        use_topology_cache=bool(scenario.get("use_topology_cache", True)),
                        base_delay=float(scenario["base_delay"]),
                        jitter=float(scenario["jitter"]),
                        message_source=int(scenario["message_source"]),
                        fanout=scenario.get("fanout"),
                        num_clusters=int(scenario["num_clusters"]),
                        ch_overload_factor=float(scenario["ch_overload_factor"]),
                        edge_prob=scenario.get("edge_prob"),
                        ba_m=scenario.get("ba_m"),
                        enable_adaptive_trace=True,
                        churn_rate=scenario.get("churn_rate"),
                        scenario_tag=f"scenario={scenario_name};max_fanout={max_fanout}",
                    )
                    adaptive = pd.DataFrame(
                        [asdict(row) for row in summary.pop("adaptive_trace_rows")]
                    )
                    if adaptive.empty:
                        raise RuntimeError("Required adaptive trace is empty")
                    adaptive.insert(0, "validation_max_fanout", max_fanout)
                    adaptive.insert(0, "validation_scenario", scenario_name)
                    trace_rows.extend(adaptive.to_dict("records"))

                    fanouts = adaptive["fanout"].astype(int)
                    raw_rows.append(
                        {
                            "scenario": scenario_name,
                            "max_fanout": max_fanout,
                            "seed": seed,
                            "run_index": run_index,
                            **{metric: summary[metric] for metric in METRICS},
                            "trace_rows": len(adaptive),
                            "mean_requested_fanout": float(fanouts.mean()),
                            "min_observed_fanout": int(fanouts.min()),
                            "max_observed_fanout": int(fanouts.max()),
                            "fanout_transition_count": int(adaptive["fanout_changed"].sum()),
                            "mode_transition_count": int(adaptive["mode_switched"].sum()),
                            "upper_bound_pct": float((fanouts == max_fanout).mean() * 100.0),
                            "fanout_above_four": bool((fanouts > 4).any()),
                            "decreased_after_above_four": decreased_after_above_four(adaptive),
                            **forwarding_diagnostics,
                        }
                    )
                    completed += 1
                    print(
                        f"[{completed:03d}/{expected_runs}] scenario={scenario_name:<10} "
                        f"max_fanout={max_fanout} seed={seed} "
                        f"delivery={summary['delivery_ratio']:.3f} "
                        f"delay={summary['propagation_delay']} duplicates={summary['duplicates']} "
                        f"requested_range={fanouts.min()}-{fanouts.max()} "
                        f"physical_excess={forwarding_diagnostics['max_physical_degree_excess']}"
                    )
    finally:
        AHBNStrategy.select_targets = original_select_targets

    raw = pd.DataFrame(raw_rows)
    trace = pd.DataFrame(trace_rows)
    raw.to_csv(RAW_PATH, index=False)
    trace.to_csv(TRACE_PATH, index=False)

    if completed != expected_runs or len(raw) != expected_runs:
        raise RuntimeError(
            f"Run-count mismatch: expected={expected_runs}, completed={completed}, rows={len(raw)}"
        )

    summary_rows: list[dict] = []
    aggregate_columns = METRICS + (
        "mean_requested_fanout",
        "min_observed_fanout",
        "max_observed_fanout",
        "fanout_transition_count",
        "mode_transition_count",
        "upper_bound_pct",
        "max_effective",
        "max_physical_degree_excess",
        "max_budget_excess",
    )
    for (scenario, max_fanout), group in raw.groupby(["scenario", "max_fanout"]):
        for metric in aggregate_columns:
            mean, ci95 = mean_ci(group[metric])
            summary_rows.append(
                {
                    "scenario": scenario,
                    "max_fanout": max_fanout,
                    "metric": metric,
                    "n": len(group),
                    "mean": mean,
                    "ci95": ci95,
                }
            )
    aggregate = pd.DataFrame(summary_rows)
    aggregate.to_csv(SUMMARY_PATH, index=False)

    above_four = bool(raw.loc[raw["max_fanout"] == 6, "fanout_above_four"].any())
    decreased = bool(raw.loc[raw["max_fanout"] == 6, "decreased_after_above_four"].any())
    permanently_six = bool(
        (raw.loc[raw["max_fanout"] == 6, "upper_bound_pct"] == 100.0).all()
    )
    physical_ok = bool((raw["max_physical_degree_excess"] <= 0).all())
    budget_ok = bool((raw["max_budget_excess"] <= 0).all())

    print("\nVALIDATION SUMMARY")
    print(f"Expected/completed runs: {expected_runs}/{completed}")
    print(f"Fanout >4 reached with max_fanout=6: {above_four}")
    print(f"Fanout subsequently decreased: {decreased}")
    print(f"Permanently at fanout 6: {permanently_six}")
    print(f"Effective forwarding respects physical degree: {physical_ok}")
    print(f"Effective forwarding respects requested budget: {budget_ok}")
    print(f"Raw results: {RAW_PATH}")
    print(f"Adaptive trace: {TRACE_PATH}")
    print(f"Aggregate summary: {SUMMARY_PATH}")

    if not physical_ok:
        raise RuntimeError("Effective forwarding exceeded active physical degree")
    if not budget_ok:
        raise RuntimeError("Effective forwarding exceeded requested controller budget")
    if not above_four or not decreased or permanently_six:
        raise RuntimeError("Required max_fanout=6 adaptive behaviour was not demonstrated")


if __name__ == "__main__":
    main()
