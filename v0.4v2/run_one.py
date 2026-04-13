from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import assign_static_clusters, build_nodes_from_graph, get_or_build_topology


def build_simulation_from_config(cfg: dict, strategy_name: str):
    seed = cfg["seed"]
    num_nodes = cfg["num_nodes"]
    topology_type = cfg["topology_type"]
    use_cache = cfg.get("use_topology_cache", True)

    base_delay = cfg.get("base_delay", 1.0)
    jitter = cfg.get("jitter", 0.2)
    message_source = cfg.get("message_source", 0)

    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_cache,
        edge_prob=cfg.get("edge_prob"),
        ba_m=cfg.get("ba_m"),
    )
    nodes = build_nodes_from_graph(graph)

    cluster_manager = None
    controller = None
    ch_overload_factor = cfg.get("ch_overload_factor", 1.0)

    if strategy_name == "gossip":
        fanout = cfg.get("fanout", 3)
        strategy = GossipStrategy(fanout=fanout)

    elif strategy_name == "cluster":
        num_clusters = cfg.get("num_clusters", 4)
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
        strategy = ClusterStrategy()

    elif strategy_name == "ahbn":
        num_clusters = cfg.get("num_clusters", 4)
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
        controller = AHBNController(AHBNParams())
        strategy = AHBNStrategy(default_fanout=cfg.get("default_fanout", 3))

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
        ch_overload_factor=ch_overload_factor,
    )
    return sim, message_source


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--strategy", required=True, choices=["gossip", "cluster", "ahbn"])
    args = parser.parse_args()

    cfg = load_yaml_config(args.config)
    sim, source_id = build_simulation_from_config(cfg, args.strategy)

    sim.inject_message(source_id=source_id, message_id="m1")
    sim.run()

    summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
    print(f"Strategy: {args.strategy}")
    print(summary)


if __name__ == "__main__":
    main()


# from __future__ import annotations

# import argparse

# from ahbn.config import load_yaml_config
# from ahbn.control import AHBNController, AHBNParams
# from ahbn.simulator import Simulator
# from ahbn.strategies.ahbn import AHBNStrategy
# from ahbn.strategies.cluster import ClusterStrategy
# from ahbn.strategies.gossip import GossipStrategy
# from ahbn.topology import assign_static_clusters, build_nodes_from_graph, build_random_graph


# def build_simulation_from_config(cfg: dict, strategy_name: str):
#     seed = cfg["seed"]
#     num_nodes = cfg["num_nodes"]
#     edge_prob = cfg["edge_prob"]
#     base_delay = cfg.get("base_delay", 1.0)
#     jitter = cfg.get("jitter", 0.2)
#     message_source = cfg.get("message_source", 0)

#     graph = build_random_graph(num_nodes=num_nodes, edge_prob=edge_prob, seed=seed)
#     nodes = build_nodes_from_graph(graph)

#     cluster_manager = None
#     controller = None
#     ch_overload_factor = cfg.get("ch_overload_factor", 1.0)

#     if strategy_name == "gossip":
#         fanout = cfg.get("fanout", 3)
#         strategy = GossipStrategy(fanout=fanout)

#     elif strategy_name == "cluster":
#         num_clusters = cfg.get("num_clusters", 4)
#         cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
#         strategy = ClusterStrategy()

#     elif strategy_name == "ahbn":
#         num_clusters = cfg.get("num_clusters", 4)
#         cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
#         controller = AHBNController(AHBNParams())
#         strategy = AHBNStrategy(default_fanout=cfg.get("default_fanout", 3))

#     else:
#         raise ValueError(f"Unknown strategy: {strategy_name}")

#     sim = Simulator(
#         nodes=nodes,
#         strategy=strategy,
#         seed=seed,
#         base_delay=base_delay,
#         jitter=jitter,
#         cluster_manager=cluster_manager,
#         controller=controller,
#         ch_overload_factor=ch_overload_factor,
#     )
#     return sim, message_source


# def main() -> None:
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--config", required=True, help="Path to YAML config")
#     parser.add_argument(
#         "--strategy",
#         required=True,
#         choices=["gossip", "cluster", "ahbn"],
#         help="Strategy to run",
#     )
#     args = parser.parse_args()

#     cfg = load_yaml_config(args.config)
#     sim, source_id = build_simulation_from_config(cfg, args.strategy)

#     sim.inject_message(source_id=source_id, message_id="m1")
#     sim.run()

#     summary = sim.metrics.summarize_message("m1", total_nodes=len(sim.nodes))
#     print(f"Strategy: {args.strategy}")
#     print(summary)


# if __name__ == "__main__":
#     main()