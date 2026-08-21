from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.churn_manager import ChurnManager
from ahbn.failure_injector import FailureInjector
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.strategies.hybrid_fixed import HybridFixedStrategy
from ahbn.topology import (
    assign_dcsoc_clusters,
    assign_mixed_resources,
    assign_static_clusters,
    build_nodes_from_graph,
    get_or_build_topology,
)
from ahbn.utils import ResultRow, save_results_csv, save_adaptive_trace_csv


def build_ahbn_params(cfg: dict) -> AHBNParams:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParams(
        alpha=ahbn_cfg.get("alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.5),
        l0=ahbn_cfg.get("l0", 0.5),
        u0=ahbn_cfg.get("u0", 0.5),
        c0=ahbn_cfg.get("c0", 0.5),
        w_d=ahbn_cfg.get("w_d", -1.0),
        w_l=ahbn_cfg.get("w_l", 1.0),
        w_u=ahbn_cfg.get("w_u", -1.0),
        w_c=ahbn_cfg.get("w_c", 1.0),
        kappa=ahbn_cfg.get("kappa", 1.0),
        beta=ahbn_cfg.get("beta", 1.0),
        min_fanout=ahbn_cfg.get("min_fanout", 2),
        max_fanout=ahbn_cfg.get("max_fanout", 4),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
    )


def build_ahbn_strategy(cfg: dict, fanout: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})
    default_fanout = (
        fanout
        if fanout is not None
        else ahbn_cfg.get("default_fanout", 3)
    )

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=True,
    )


def run_single(
    cfg: dict,
    strategy_name: str,
    seed: int,
    topology_type: str,
    num_nodes: int,
    use_topology_cache: bool,
    base_delay: float,
    jitter: float,
    message_source: int,
    fanout: int | None = None,
    num_clusters: int | None = None,
    ch_overload_factor: float | None = None,
    edge_prob: float | None = None,
    ba_m: int | None = None,
    failure_mode: str | None = None,
    enable_adaptive_trace: bool = False,
    churn_rate: float | None = None,
    resource_scenario: str | None = None,
    scenario_tag: str | None = None,
) -> dict:
    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_topology_cache,
        edge_prob=edge_prob,
        ba_m=ba_m,
    )
    nodes = build_nodes_from_graph(graph)

    experiment_name = cfg.get("experiment", "")
    fulfill_all_obligations = experiment_name in {"exp08", "exp09"}
    if experiment_name == "exp12":
        assign_mixed_resources(nodes, cfg, seed=seed, scenario_name=resource_scenario)

    cluster_manager = None
    controller = None

    if strategy_name == "gossip":
        strategy = GossipStrategy(
            fanout=(
                None
                if fulfill_all_obligations
                else (fanout if fanout is not None else 3)
            )
        )

    elif strategy_name == "cluster":
        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
            resource_aware_heads=False,
        )
        strategy = ClusterStrategy()

    elif strategy_name == "ahbn":
        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
            resource_aware_heads=False,
        )
        controller = AHBNController(build_ahbn_params(cfg))
        strategy = build_ahbn_strategy(cfg, fanout=fanout)

    elif strategy_name == "hybrid_fixed":

        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters or 4,
        )

        strategy = HybridFixedStrategy(
            fanout=(
                fanout
                if fanout is not None
                else 3
            )
        )

    elif strategy_name == "dcsoc":

        dcsoc_cfg = cfg.get(
            "dcsoc",
            {},
        )

        cluster_manager = assign_dcsoc_clusters(
            nodes,
            eps=float(
                dcsoc_cfg.get(
                    "eps",
                    2.0,
                )
            ),
            min_samples=int(
                dcsoc_cfg.get(
                    "min_samples",
                    3,
                )
            ),
        )

        strategy = DCSOCStrategy(
            fanout=int(
                dcsoc_cfg.get(
                    "fanout",
                    (
                        fanout
                        if fanout is not None
                        else 3
                    ),
                )
            ),
            inter_fanout=int(
                dcsoc_cfg.get(
                    "inter_fanout",
                    1,
                )
            ),
            fulfill_all_structural_children=fulfill_all_obligations,
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

    local_cfg = dict(cfg)
    if failure_mode is not None:
        local_failure = dict(cfg.get("failure", {}))
        local_failure["mode"] = failure_mode
        local_cfg["failure"] = local_failure

    if churn_rate is not None:
        local_churn = dict(cfg.get("churn", {}))
        local_churn["target_fraction"] = churn_rate
        local_cfg["churn"] = local_churn

    failure_injector = FailureInjector(local_cfg, seed=seed)
    churn_manager = ChurnManager(local_cfg, seed=seed)

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=cluster_manager,
        controller=controller,
        ch_overload_factor=ch_overload_factor if ch_overload_factor is not None else 1.0,
        failure_injector=failure_injector,
        churn_manager=churn_manager,
        experiment_name=cfg.get("experiment", "unknown"),
        strategy_name=strategy_name,
        scenario_tag=(
            scenario_tag
            if scenario_tag is not None
            else (
                resource_scenario
                if resource_scenario is not None
                else (failure_mode if failure_mode is not None else topology_type)
            )
        ),
        enable_adaptive_trace=enable_adaptive_trace,
        resource_aware_heads=False,
    )

    sim.inject_message(source_id=message_source, message_id="m1")
    sim.run()

    summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
    summary.update(sim.get_resource_metrics())
    if enable_adaptive_trace:
        summary["adaptive_trace_rows"] = sim.adaptive_trace_rows
    return summary


def exp07(cfg: dict) -> tuple[list[ResultRow], list]:
    rows: list[ResultRow] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    fanouts = cfg["fanouts"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "ahbn"])

    if "gossip" in strategies:
        for fanout in fanouts:
            for run_idx in range(runs_per_setting):
                seed = base_seed + run_idx

                summary = run_single(
                    cfg=cfg,
                    strategy_name="gossip",
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    enable_adaptive_trace=False,
                    scenario_tag=f"fanout={fanout}",
                )
                rows.append(
                    ResultRow(
                        experiment="exp07",
                        strategy="gossip",
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type=topology_type,
                        topology_param=edge_prob if topology_type == "er" else ba_m,
                        fanout=fanout,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )

    if "ahbn" in strategies:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            summary = run_single(
                cfg=cfg,
                strategy_name="ahbn",
                seed=seed,
                topology_type=topology_type,
                num_nodes=num_nodes,
                use_topology_cache=use_topology_cache,
                base_delay=base_delay,
                jitter=jitter,
                message_source=source_id,
                fanout=None,
                num_clusters=num_clusters,
                edge_prob=edge_prob,
                ba_m=ba_m,
                enable_adaptive_trace=True,
                scenario_tag="adaptive",
            )
            rows.append(
                ResultRow(
                    experiment="exp07",
                    strategy="ahbn",
                    seed=seed,
                    num_nodes=num_nodes,
                    topology_type=topology_type,
                    topology_param=edge_prob if topology_type == "er" else ba_m,
                    fanout=None,
                    num_clusters=num_clusters,
                    ch_overload_factor=None,
                    delivery_ratio=summary["delivery_ratio"],
                    propagation_delay=summary["propagation_delay"],
                    duplicates=summary["duplicates"],
                    total_forwards=summary["total_forwards"],
                )
            )

            if "adaptive_trace_rows" in summary:
                trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp08(cfg: dict) -> tuple[list[ResultRow], list]:
    rows: list[ResultRow] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    overload_values = cfg["ch_overload_factor"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    num_clusters = cfg["num_clusters"]

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["cluster", "ahbn"])

    for overload in overload_values:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    num_clusters=num_clusters,
                    ch_overload_factor=overload,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                    scenario_tag=f"ch_overload_factor={overload}",
                )
                rows.append(
                    ResultRow(
                        experiment="exp08",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type=topology_type,
                        topology_param=edge_prob if topology_type == "er" else ba_m,
                        fanout=None,
                        num_clusters=num_clusters,
                        ch_overload_factor=overload,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp09(cfg: dict) -> tuple[list[ResultRow], list]:
    rows: list[ResultRow] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    if topology_type != "er":
        raise ValueError("Exp09 density sweep is intended for ER topology.")

    edge_probs = cfg["edge_probs"]
    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])

    for edge_prob in edge_probs:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type="er",
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                    scenario_tag=f"edge_prob={edge_prob}",
                )
                rows.append(
                    ResultRow(
                        experiment="exp09",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type="er",
                        topology_param=edge_prob,
                        fanout=None,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp10(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])
    failure_modes = cfg.get("failure_modes", ["node_failure", "ch_failure", "overload"])

    for failure_mode in failure_modes:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    failure_mode=failure_mode,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                )

                rows.append(
                    {
                        "experiment": "exp10",
                        "strategy": strategy_name,
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "topology_type": topology_type,
                        "topology_param": edge_prob if topology_type == "er" else ba_m,
                        "fanout": fanout if strategy_name != "cluster" else None,
                        "num_clusters": num_clusters,
                        "ch_overload_factor": None,
                        "failure_mode": summary["failure_mode"],
                        "failed_node_id": summary["failed_node_id"],
                        "delivery_ratio": summary["delivery_ratio"],
                        "propagation_delay": summary["propagation_delay"],
                        "duplicates": summary["duplicates"],
                        "total_forwards": summary["total_forwards"],
                        "recovery_time": summary["recovery_time"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp11(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])
    churn_rates = cfg.get("churn_rates", [0.0, 0.05, 0.10, 0.20, 0.30])

    for churn_rate in churn_rates:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    churn_rate=churn_rate,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                )

                rows.append(
                    {
                        "experiment": "exp11",
                        "strategy": strategy_name,
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "topology_type": topology_type,
                        "topology_param": edge_prob if topology_type == "er" else ba_m,
                        "fanout": fanout if strategy_name != "cluster" else None,
                        "num_clusters": num_clusters,
                        "churn_rate": churn_rate,
                        "delivery_ratio": summary["delivery_ratio"],
                        "propagation_delay": summary["propagation_delay"],
                        "duplicates": summary["duplicates"],
                        "total_forwards": summary["total_forwards"],
                        "churn_event_count": summary["churn_event_count"],
                        "churn_leave_count": summary["churn_leave_count"],
                        "churn_join_count": summary["churn_join_count"],
                        "cluster_repair_count": summary["cluster_repair_count"],
                        "mode_switch_count": summary["mode_switch_count"],
                        "fanout_change_count": summary["fanout_change_count"],
                        "adaptation_event_count": summary["adaptation_event_count"],
                        "adaptation_rate": summary["adaptation_rate"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def exp12(cfg: dict) -> tuple[list[dict], list]:
    rows: list[dict] = []
    trace_rows: list = []

    base_seed = cfg["seed"]
    runs_per_setting = cfg["runs_per_setting"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_topology_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    source_id = cfg.get("message_source", 0)
    fanout = cfg.get("fanout", 3)
    num_clusters = cfg.get("num_clusters", 4)

    edge_prob = cfg.get("edge_prob")
    ba_m = cfg.get("ba_m")

    strategies = cfg.get("strategies", ["gossip", "cluster", "ahbn"])
    resource_scenarios = cfg.get("resource_scenarios", ["balanced", "weak_heavy"])

    for resource_scenario in resource_scenarios:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in strategies:
                summary = run_single(
                    cfg=cfg,
                    strategy_name=strategy_name,
                    seed=seed,
                    topology_type=topology_type,
                    num_nodes=num_nodes,
                    use_topology_cache=use_topology_cache,
                    base_delay=base_delay,
                    jitter=jitter,
                    message_source=source_id,
                    fanout=fanout,
                    num_clusters=num_clusters,
                    edge_prob=edge_prob,
                    ba_m=ba_m,
                    resource_scenario=resource_scenario,
                    enable_adaptive_trace=(strategy_name == "ahbn"),
                )

                rows.append(
                    {
                        "experiment": "exp12",
                        "strategy": strategy_name,
                        "seed": seed,
                        "num_nodes": num_nodes,
                        "topology_type": topology_type,
                        "topology_param": edge_prob if topology_type == "er" else ba_m,
                        "fanout": fanout if strategy_name != "cluster" else None,
                        "num_clusters": num_clusters,
                        "resource_scenario": resource_scenario,
                        "delivery_ratio": summary["delivery_ratio"],
                        "propagation_delay": summary["propagation_delay"],
                        "duplicates": summary["duplicates"],
                        "total_forwards": summary["total_forwards"],
                        "max_normalized_load": summary["max_normalized_load"],
                        "load_balance_cv": summary["load_balance_cv"],
                        "strong_forward_share": summary["strong_forward_share"],
                        "medium_forward_share": summary["medium_forward_share"],
                        "weak_forward_share": summary["weak_forward_share"],
                    }
                )

                if "adaptive_trace_rows" in summary:
                    trace_rows.extend(summary["adaptive_trace_rows"])

    return rows, trace_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    experiment = cfg["experiment"]

    if experiment == "exp07":
        rows, trace_rows = exp07(cfg)
        path = save_results_csv(rows, "outputs/csv/exp07_results.csv")
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp07_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp08":
        rows, trace_rows = exp08(cfg)
        path = save_results_csv(rows, "outputs/csv/exp08_results.csv")
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp08_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp09":
        rows, trace_rows = exp09(cfg)
        path = save_results_csv(rows, "outputs/csv/exp09_results.csv")
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp09_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp10":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp10(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp10_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp10_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp11":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp11(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp11_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp11_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    elif experiment == "exp12":
        import pandas as pd
        from pathlib import Path
        from ahbn.utils import current_timestamp

        rows, trace_rows = exp12(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        ts = current_timestamp()
        path = out / f"exp12_results_{ts}.csv"
        pd.DataFrame(rows).to_csv(path, index=False)
        print(f"Saved {path}")

        if trace_rows:
            trace_path = save_adaptive_trace_csv(
                trace_rows,
                "outputs/csv/exp12_adaptive_trace.csv",
                add_timestamp=True,
            )
            print(f"Saved {trace_path}")

    else:
        raise ValueError(f"Unsupported experiment: {experiment}")


if __name__ == "__main__":
    main()
