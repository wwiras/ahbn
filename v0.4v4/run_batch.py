from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.failure_injector import FailureInjector
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.strategies.hybrid_fixed import HybridFixedStrategy
from ahbn.topology import (
    assign_static_clusters,
    build_nodes_from_graph,
    get_or_build_topology,
)
from ahbn.utils import ResultRow, save_results_csv, save_adaptive_trace_csv


def build_ahbn_params(cfg: dict) -> AHBNParams:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParams(
        ewma_alpha=ahbn_cfg.get("ewma_alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.2),
        u0=ahbn_cfg.get("u0", 5.0),
        l0=ahbn_cfg.get("l0", 2.0),
        rho0=ahbn_cfg.get("rho0", 0.1),
        deg0=ahbn_cfg.get("deg0", 8.0),
        ov0=ahbn_cfg.get("ov0", 0.25),
        r0=ahbn_cfg.get("r0", 0.35),
        a_dup=ahbn_cfg.get("a_dup", -2.0),
        a_load=ahbn_cfg.get("a_load", -1.5),
        a_lat=ahbn_cfg.get("a_lat", 1.5),
        a_churn=ahbn_cfg.get("a_churn", 1.0),
        a_deg=ahbn_cfg.get("a_deg", -0.4),
        a_ov=ahbn_cfg.get("a_ov", -1.2),
        a_red=ahbn_cfg.get("a_red", -1.8),
        b_degree=ahbn_cfg.get("b_degree", 0.25),
        b_overlap=ahbn_cfg.get("b_overlap", 0.75),
        min_fanout=ahbn_cfg.get("min_fanout", 1),
        max_fanout=ahbn_cfg.get("max_fanout", 6),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
        fanout_dup_penalty=ahbn_cfg.get("fanout_dup_penalty", 2.0),
        fanout_load_penalty=ahbn_cfg.get("fanout_load_penalty", 0.5),
        fanout_lat_reward=ahbn_cfg.get("fanout_lat_reward", 0.8),
        fanout_red_penalty=ahbn_cfg.get("fanout_red_penalty", 1.5),
        tau_max=ahbn_cfg.get("tau_max", 0.90),
        tau_min=ahbn_cfg.get("tau_min", 0.25),
        tau_dup_penalty=ahbn_cfg.get("tau_dup_penalty", 1.0),
        tau_red_penalty=ahbn_cfg.get("tau_red_penalty", 1.5),
        min_weight=ahbn_cfg.get("min_weight", 0.20),
        max_weight=ahbn_cfg.get("max_weight", 0.80),
    )


def build_ahbn_strategy(cfg: dict, fanout: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})
    default_fanout = fanout if fanout is not None else ahbn_cfg.get("default_fanout", 3)

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=ahbn_cfg.get("adaptive_fanout", False),
        hybrid_mode=ahbn_cfg.get("hybrid_mode", True),
        use_tau_gate=ahbn_cfg.get("use_tau_gate", True),
        min_cluster_targets=ahbn_cfg.get("min_cluster_targets", 1),
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

    cluster_manager = None
    controller = None

    if strategy_name == "gossip":
        strategy = GossipStrategy(fanout=fanout if fanout is not None else 3)

    elif strategy_name == "cluster":
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters or 4)
        strategy = ClusterStrategy()

    elif strategy_name == "ahbn":
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters or 4)
        controller = AHBNController(build_ahbn_params(cfg))
        strategy = build_ahbn_strategy(cfg, fanout=fanout)

    elif strategy_name == "hybrid_fixed":
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters or 4)
        strategy = HybridFixedStrategy(fanout=fanout if fanout is not None else 3)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    local_cfg = dict(cfg)
    if failure_mode is not None:
        local_failure = dict(cfg.get("failure", {}))
        local_failure["mode"] = failure_mode
        local_cfg["failure"] = local_failure

    failure_injector = FailureInjector(local_cfg, seed=seed)

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
        experiment_name=cfg.get("experiment", "unknown"),
        strategy_name=strategy_name,
        scenario_tag=failure_mode if failure_mode is not None else topology_type,
        enable_adaptive_trace=enable_adaptive_trace,
    )

    sim.inject_message(source_id=message_source, message_id="m1")
    sim.run()

    summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
    if enable_adaptive_trace:
        summary["adaptive_trace_rows"] = sim.adaptive_trace_rows
    return summary


def exp07(cfg: dict) -> list[ResultRow]:
    rows: list[ResultRow] = []

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

    for fanout in fanouts:
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
                )
                rows.append(
                    ResultRow(
                        experiment="exp07",
                        strategy=strategy_name,
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
    return rows


def exp08(cfg: dict) -> list[ResultRow]:
    rows: list[ResultRow] = []

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
    return rows


def exp09(cfg: dict) -> list[ResultRow]:
    rows: list[ResultRow] = []

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
                )
                rows.append(
                    ResultRow(
                        experiment="exp09",
                        strategy=strategy_name,
                        seed=seed,
                        num_nodes=num_nodes,
                        topology_type="er",
                        topology_param=edge_prob,
                        fanout=fanout if strategy_name != "cluster" else None,
                        num_clusters=num_clusters,
                        ch_overload_factor=None,
                        delivery_ratio=summary["delivery_ratio"],
                        propagation_delay=summary["propagation_delay"],
                        duplicates=summary["duplicates"],
                        total_forwards=summary["total_forwards"],
                    )
                )
    return rows



    rows: list[dict] = []

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

    return rows


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    experiment = cfg["experiment"]

    if experiment == "exp07":
        rows = exp07(cfg)
        path = save_results_csv(rows, "outputs/csv/exp07_results.csv")
        print(f"Saved {path}")

    elif experiment == "exp08":
        rows = exp08(cfg)
        path = save_results_csv(rows, "outputs/csv/exp08_results.csv")
        print(f"Saved {path}")

    elif experiment == "exp09":
        rows = exp09(cfg)
        path = save_results_csv(rows, "outputs/csv/exp09_results.csv")
        print(f"Saved {path}")

    elif experiment == "exp10":
        import pandas as pd
        from pathlib import Path

        rows, trace_rows = exp10(cfg)
        out = Path("outputs/csv")
        out.mkdir(parents=True, exist_ok=True)

        from ahbn.utils import current_timestamp
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

    else:
        raise ValueError(f"Unsupported experiment: {experiment}")


if __name__ == "__main__":
    main()