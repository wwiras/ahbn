from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.strategies.hybrid_fixed import HybridFixedStrategy
from ahbn.topology import assign_static_clusters, build_nodes_from_graph, get_or_build_topology
from ahbn.utils import ResultRow, save_results_csv


def run_single(
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
    adaptive_fanout: bool = False,
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
        controller = AHBNController(AHBNParams())
        strategy = AHBNStrategy(
            default_fanout=fanout if fanout is not None else 3,
            adaptive_fanout=adaptive_fanout,
        )
    elif strategy_name == "hybrid_fixed":
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters or 4)
        strategy = HybridFixedStrategy(fanout=fanout if fanout is not None else 3)
    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=cluster_manager,
        controller=controller,
        ch_overload_factor=ch_overload_factor if ch_overload_factor is not None else 1.0,
    )

    sim.inject_message(source_id=message_source, message_id="m1")
    sim.run()
    return sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))


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

    # Exp07 is a fanout sensitivity experiment.
    # Therefore AHBN should respect the configured fanout directly.
    adaptive_fanout = cfg.get("adaptive_fanout", False)

    for fanout in fanouts:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            # for strategy_name in ["gossip", "ahbn"]:
            for strategy_name in ["gossip", "hybrid_fixed"]:
                summary = run_single(
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
                    adaptive_fanout=adaptive_fanout if strategy_name == "ahbn" else False,
                )
                rows.append(
                    ResultRow(
                        experiment="exp07",
                        # strategy=strategy_name,
                        strategy="ahbn" if strategy_name == "hybrid_fixed" else strategy_name,
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

    # For Exp08, adaptive fanout may be useful later if you want AHBN to behave adaptively.
    adaptive_fanout = cfg.get("adaptive_fanout", True)

    for overload in overload_values:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in ["cluster", "ahbn"]:
                summary = run_single(
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
                    adaptive_fanout=adaptive_fanout if strategy_name == "ahbn" else False,
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

    # For Exp09, you can choose whether AHBN should remain fixed-fanout or adaptive.
    adaptive_fanout = cfg.get("adaptive_fanout", False)

    for edge_prob in edge_probs:
        for run_idx in range(runs_per_setting):
            seed = base_seed + run_idx

            for strategy_name in ["gossip", "cluster", "ahbn"]:
                summary = run_single(
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
                    adaptive_fanout=adaptive_fanout if strategy_name == "ahbn" else False,
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

    else:
        raise ValueError(f"Unknown experiment: {experiment}")


if __name__ == "__main__":
    main()