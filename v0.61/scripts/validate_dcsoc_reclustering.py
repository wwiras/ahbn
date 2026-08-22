"""S2 deterministic du validation: equivalent and changed network states."""
from __future__ import annotations

import sys
import networkx as nx

from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph, recluster_dcsoc


def snapshot(nodes, manager):
    return ({nid: node.cluster_id for nid, node in nodes.items()}, tuple(manager.structural_edges))


def main() -> int:
    graph = nx.Graph()
    graph.add_edges_from(list(nx.complete_graph(range(4)).edges()) +
                         list(nx.complete_graph(range(4, 8)).edges()) + [(3, 8), (8, 9), (9, 4)])
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, 1.0, 4)
    initial = snapshot(nodes, manager)
    recluster_dcsoc(nodes, manager, 1.0, 4)
    equivalent = snapshot(nodes, manager) == initial and manager.recluster_count == 1

    # Genuine network-state change: dense cross-links merge the two density regions.
    for left in range(4):
        for right in range(4, 8):
            if right not in nodes[left].original_neighbors:
                nodes[left].original_neighbors.append(right)
                nodes[right].original_neighbors.append(left)
    before_changed = snapshot(nodes, manager)
    recluster_dcsoc(nodes, manager, 1.0, 4)
    changed = snapshot(nodes, manager) != before_changed and manager.recluster_count == 2
    deterministic = []
    for _ in range(2):
        clone = build_nodes_from_graph(graph)
        deterministic.append(snapshot(clone, assign_dcsoc_clusters(clone, 1.0, 4)))
    checks = {"equivalent update stays equivalent": equivalent,
              "changed inputs regenerate structure": changed,
              "fixed-input determinism": deterministic[0] == deterministic[1]}
    for label, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} — {label}")
    print(f"recluster_count={manager.recluster_count} generation={manager.structural_generation} "
          f"topology_edges_changed={manager.topology_edges_changed}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
