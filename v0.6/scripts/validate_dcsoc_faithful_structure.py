"""S2 pre/post validator for explicit DC-SoC propagation structure."""
from __future__ import annotations

import sys

import networkx as nx

from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


def main() -> int:
    graph = nx.Graph()
    graph.add_edges_from(
        [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
         (3, 4), (4, 5),
         (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
         (5, 6)]
    )
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=1.0, min_samples=4)

    required_node_fields = ("dcsoc_role", "dcsoc_parent", "dcsoc_children", "dcsoc_core_neighbors")
    required_manager_fields = ("structural_edges", "structural_generation", "initial_clustering_count")
    missing = [name for name in required_node_fields if not all(hasattr(n, name) for n in nodes.values())]
    missing += [name for name in required_manager_fields if not hasattr(manager, name)]
    if missing:
        print("FAIL — frozen comparator has no explicit faithful structure: " + ", ".join(missing))
        return 1

    edges = list(manager.structural_edges)
    dag = nx.DiGraph(edges)
    active = {nid for nid, node in nodes.items() if node.is_active}
    covered = set(dag.nodes) | ({next(iter(active))} if len(active) == 1 else set())
    checks = {
        "all active nodes covered": covered == active,
        "acyclic": nx.is_directed_acyclic_graph(dag),
        "no self loops": all(src != dst for src, dst in edges),
        "no duplicate edges": len(edges) == len(set(edges)),
        "reciprocal parent/children": all(
            nodes[child].dcsoc_parent == parent and child in nodes[parent].dcsoc_children
            for parent, child in edges
        ),
        "at least two cores": sum(n.dcsoc_role == "core" for n in nodes.values()) >= 2,
        "ordinary leaves present": any(n.dcsoc_role == "leaf" for n in nodes.values()),
    }
    for label, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} — {label}")
    if not all(checks.values()):
        return 1
    print("PASS — explicit DC-SoC propagation structure is valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
