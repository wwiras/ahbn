"""S2 validator: leaves may uplink, but may not independently fan out."""
from __future__ import annotations

import random
import sys
from types import SimpleNamespace

import networkx as nx

from ahbn.message import Message
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


def main() -> int:
    graph = nx.complete_graph(6)
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=2.0, min_samples=3)
    leaf = next(node for node in nodes.values() if node.dcsoc_role == "leaf")
    sim = SimpleNamespace(nodes=nodes, cluster_manager=manager, rng=random.Random(42))
    targets = DCSOCStrategy(fanout=3, inter_fanout=1).select_targets(
        leaf, Message("s2-leaf", leaf.node_id, 0.0), sim
    )
    allowed = [] if leaf.dcsoc_parent is None else [leaf.dcsoc_parent]
    ok = targets == allowed
    print(f"leaf={leaf.node_id} parent={leaf.dcsoc_parent} targets={targets}")
    print(("PASS" if ok else "FAIL") + " — ordinary leaf does not independently fan out")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
