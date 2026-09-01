from __future__ import annotations

import argparse

from ahbn.config import load_yaml_config
from ahbn.control import AHBNController, AHBNParams
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import (
    assign_dcsoc_clusters,
    assign_static_clusters,
    build_nodes_from_graph,
    get_or_build_topology,
)


def build_ahbn_params(cfg: dict) -> AHBNParams:
    ahbn_cfg = cfg.get("ahbn", {})

    return AHBNParams(
        alpha=ahbn_cfg.get("alpha", 0.3),
        d0=ahbn_cfg.get("d0", 0.0),
        l0=ahbn_cfg.get("l0", 0.0),
        u0=ahbn_cfg.get("u0", 0.0),
        c0=ahbn_cfg.get("c0", 0.0),
        w_d=ahbn_cfg.get("w_d", -1.0),
        w_l=ahbn_cfg.get("w_l", 1.0),
        w_u=ahbn_cfg.get("w_u", 1.0),
        w_c=ahbn_cfg.get("w_c", 1.0),
        kappa=ahbn_cfg.get("kappa", 1.0),
        beta=ahbn_cfg.get("beta", 1.0),
        min_fanout=ahbn_cfg.get("min_fanout", 2),
        max_fanout=ahbn_cfg.get("max_fanout", 6),
        mode_threshold=ahbn_cfg.get("mode_threshold", 0.5),
    )


def build_ahbn_strategy(cfg: dict, fanout_override: int | None = None) -> AHBNStrategy:
    ahbn_cfg = cfg.get("ahbn", {})

    default_fanout = (
        fanout_override
        if fanout_override is not None
        else ahbn_cfg.get("default_fanout", 3)
    )

    return AHBNStrategy(
        default_fanout=default_fanout,
        adaptive_fanout=True,
    )


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
        strategy = GossipStrategy(fanout=cfg.get("fanout"))

    elif strategy_name == "cluster":
        num_clusters = cfg.get("num_clusters", 4)
        cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
        strategy = ClusterStrategy()

    # elif strategy_name == "ahbn":
    #     num_clusters = cfg.get("num_clusters", 4)
    #     cluster_manager = assign_static_clusters(nodes, num_clusters=num_clusters)
    #     controller = AHBNController(build_ahbn_params(cfg))
    #     strategy = build_ahbn_strategy(cfg, fanout_override=cfg.get("fanout"))

    # else:
    #     raise ValueError(f"Unknown strategy: {strategy_name}")
    
    elif strategy_name == "ahbn":
        num_clusters = cfg.get("num_clusters", 4)

        cluster_manager = assign_static_clusters(
            nodes,
            num_clusters=num_clusters,
        )

        controller = AHBNController(
            build_ahbn_params(cfg)
        )

        strategy = build_ahbn_strategy(
            cfg,
            fanout_override=cfg.get("fanout"),
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

        strategy = DCSOCStrategy()

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}"
        )

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
    # parser.add_argument("--strategy", required=True, choices=["gossip", "cluster", "ahbn"])
    parser.add_argument(
        "--strategy",
        required=True,
        choices=[
            "gossip",
            "cluster",
            "dcsoc",
            "ahbn",
        ],
    )
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
