from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Dict

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN

from ahbn.cluster import ClusterManager
from ahbn.node import Node


TOPOLOGY_CACHE_DIR = Path("outputs/topologies")


def ensure_cache_dir() -> None:
    TOPOLOGY_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def relabel_graph_compact(g: nx.Graph) -> nx.Graph:
    mapping = {old: new for new, old in enumerate(sorted(g.nodes()))}
    return nx.relabel_nodes(g, mapping)


def largest_connected_subgraph(g: nx.Graph) -> nx.Graph:
    if g.number_of_nodes() == 0:
        raise ValueError("Generated graph has zero nodes.")
    if nx.is_connected(g):
        return g.copy()
    largest = max(nx.connected_components(g), key=len)
    return g.subgraph(largest).copy()


def build_er_graph(num_nodes: int, edge_prob: float, seed: int) -> nx.Graph:
    rng = random.Random(seed)
    graph_seed = rng.randint(0, 10_000_000)
    g = nx.erdos_renyi_graph(num_nodes, edge_prob, seed=graph_seed)
    g = largest_connected_subgraph(g)
    g = relabel_graph_compact(g)
    return g


def build_ba_graph(num_nodes: int, m: int, seed: int) -> nx.Graph:
    if m <= 0:
        raise ValueError("BA parameter m must be > 0")
    if m >= num_nodes:
        raise ValueError("BA parameter m must be less than num_nodes")

    rng = random.Random(seed)
    graph_seed = rng.randint(0, 10_000_000)
    g = nx.barabasi_albert_graph(num_nodes, m, seed=graph_seed)
    g = largest_connected_subgraph(g)
    g = relabel_graph_compact(g)
    return g


def topology_cache_path(topology_type: str, num_nodes: int, param_name: str, param_value: float | int, seed: int) -> Path:
    filename = f"{topology_type}_n{num_nodes}_{param_name}{param_value}_seed{seed}.json"
    return TOPOLOGY_CACHE_DIR / filename


def save_graph_to_cache(graph: nx.Graph, path: Path) -> None:
    data = {
        "nodes": sorted(graph.nodes()),
        "edges": sorted([sorted([u, v]) for u, v in graph.edges()]),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def load_graph_from_cache(path: Path) -> nx.Graph:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    g = nx.Graph()
    g.add_nodes_from(data["nodes"])
    g.add_edges_from([tuple(edge) for edge in data["edges"]])
    return g


def get_or_build_topology(
    topology_type: str,
    num_nodes: int,
    seed: int,
    use_cache: bool = True,
    edge_prob: float | None = None,
    ba_m: int | None = None,
) -> nx.Graph:
    ensure_cache_dir()

    if topology_type == "er":
        if edge_prob is None:
            raise ValueError("edge_prob is required for ER topology")
        cache_path = topology_cache_path("er", num_nodes, "p", edge_prob, seed)

        if use_cache and cache_path.exists():
            return load_graph_from_cache(cache_path)

        graph = build_er_graph(num_nodes=num_nodes, edge_prob=edge_prob, seed=seed)

        if use_cache:
            save_graph_to_cache(graph, cache_path)

        return graph

    if topology_type == "ba":
        if ba_m is None:
            raise ValueError("ba_m is required for BA topology")
        cache_path = topology_cache_path("ba", num_nodes, "m", ba_m, seed)

        if use_cache and cache_path.exists():
            return load_graph_from_cache(cache_path)

        graph = build_ba_graph(num_nodes=num_nodes, m=ba_m, seed=seed)

        if use_cache:
            save_graph_to_cache(graph, cache_path)

        return graph

    raise ValueError(f"Unsupported topology_type: {topology_type}")


def build_nodes_from_graph(graph: nx.Graph) -> Dict[int, Node]:
    nodes: Dict[int, Node] = {}
    for n in sorted(graph.nodes()):
        nodes[n] = Node(node_id=n, neighbors=list(sorted(graph.neighbors(n))))
    return nodes


def assign_mixed_resources(nodes: Dict[int, Node], cfg: dict, seed: int, scenario_name: str | None = None) -> None:
    resources_cfg = cfg.get("resources", {})
    classes_cfg = resources_cfg.get("classes", {})
    profiles_cfg = resources_cfg.get("profiles", {})

    if not classes_cfg:
        return

    if scenario_name is None:
        fractions = resources_cfg.get("fractions", {"strong": 0.2, "medium": 0.5, "weak": 0.3})
    else:
        fractions = profiles_cfg.get(scenario_name)
        if fractions is None:
            raise ValueError(f"Unknown resource scenario: {scenario_name}")

    node_ids = sorted(nodes.keys())
    rng = random.Random(seed)
    rng.shuffle(node_ids)

    total = len(node_ids)
    remaining_ids = node_ids[:]
    allocated: dict[str, list[int]] = {}
    classes = list(fractions.keys())

    for idx, cls_name in enumerate(classes):
        if idx == len(classes) - 1:
            selected = remaining_ids[:]
        else:
            count = int(round(float(fractions.get(cls_name, 0.0)) * total))
            count = max(0, min(count, len(remaining_ids)))
            selected = remaining_ids[:count]
            remaining_ids = remaining_ids[count:]
        allocated[cls_name] = selected

    for cls_name, ids in allocated.items():
        cls_cfg = classes_cfg.get(cls_name, {})
        processing_delay = float(cls_cfg.get("processing_delay", 0.0))
        capacity_score = float(cls_cfg.get("capacity_score", 1.0))
        for node_id in ids:
            node = nodes[node_id]
            node.resource_class = cls_name
            node.processing_delay = processing_delay
            node.capacity_score = capacity_score


def _select_cluster_head(member_ids: list[int], nodes: Dict[int, Node], resource_aware_heads: bool) -> int:
    if not resource_aware_heads:
        return min(member_ids)
    return max(
        member_ids,
        key=lambda nid: (
            nodes[nid].capacity_score,
            len(nodes[nid].neighbors),
            -nid,
        ),
    )


def assign_static_clusters(
    nodes: Dict[int, Node],
    num_clusters: int,
    resource_aware_heads: bool = False,
) -> ClusterManager:
    if num_clusters <= 0:
        raise ValueError("num_clusters must be > 0")

    node_ids = sorted(nodes.keys())
    cluster_mgr = ClusterManager()

    for idx, node_id in enumerate(node_ids):
        cluster_id = idx % num_clusters
        nodes[node_id].cluster_id = cluster_id
        cluster_mgr.cluster_to_members.setdefault(cluster_id, []).append(node_id)

    for cluster_id, members in cluster_mgr.cluster_to_members.items():
        if cluster_mgr.head_selection == "highest_physical_degree":
            head_id = max(
                members,
                key=lambda nid: (
                    len(nodes[nid].original_neighbors),
                    -nid,
                ),
            )
        else:
            head_id = _select_cluster_head(members, nodes, resource_aware_heads)
        cluster_mgr.cluster_to_head[cluster_id] = head_id
        nodes[head_id].is_cluster_head = True

    cluster_ids = sorted(cluster_mgr.cluster_to_head.keys())
    for i in range(len(cluster_ids) - 1):
        left = cluster_mgr.cluster_to_head[cluster_ids[i]]
        right = cluster_mgr.cluster_to_head[cluster_ids[i + 1]]
        nodes[left].gateway_neighbors.append(right)
        nodes[right].gateway_neighbors.append(left)

    return cluster_mgr


def refresh_active_neighbors(nodes: Dict[int, Node]) -> None:
    for node in nodes.values():
        if not node.is_active:
            node.neighbors = []
            continue
        node.neighbors = [
            nbr_id
            for nbr_id in node.original_neighbors
            if nbr_id in nodes and nodes[nbr_id].is_active
        ]


def refresh_cluster_overlay(
    nodes: Dict[int, Node],
    cluster_mgr: ClusterManager | None,
    resource_aware_heads: bool = False,
) -> None:
    if cluster_mgr is None:
        return

    for node in nodes.values():
        node.is_cluster_head = False
        node.gateway_neighbors = []

    cluster_mgr.cluster_to_members = {}
    cluster_mgr.cluster_to_head = {}

    for node in nodes.values():
        if not node.is_active or node.cluster_id is None:
            continue
        cluster_mgr.cluster_to_members.setdefault(node.cluster_id, []).append(node.node_id)

    for cluster_id, members in cluster_mgr.cluster_to_members.items():
        members = sorted(members)
        cluster_mgr.cluster_to_members[cluster_id] = members
        if not members:
            continue
        head_id = _select_cluster_head(members, nodes, resource_aware_heads)
        cluster_mgr.cluster_to_head[cluster_id] = head_id
        nodes[head_id].is_cluster_head = True

    cluster_ids = sorted(cluster_mgr.cluster_to_head.keys())
    for i in range(len(cluster_ids) - 1):
        left = cluster_mgr.cluster_to_head[cluster_ids[i]]
        right = cluster_mgr.cluster_to_head[cluster_ids[i + 1]]
        nodes[left].gateway_neighbors.append(right)
        nodes[right].gateway_neighbors.append(left)


def repair_topology_after_churn(
    nodes: Dict[int, Node],
    cluster_mgr: ClusterManager | None,
    resource_aware_heads: bool = False,
) -> None:
    refresh_active_neighbors(nodes)
    refresh_cluster_overlay(nodes, cluster_mgr, resource_aware_heads=resource_aware_heads)


def _build_dcsoc_structure(nodes: Dict[int, Node], cluster_mgr: ClusterManager) -> None:
    """Build a deterministic core-rooted dissemination DAG.

    One elected dissemination core represents each retained DBSCAN cluster.
    Cluster cores form a directed routing chain and every ordinary member is
    a direct leaf of its core.  This is the documented dissemination-only
    surrogate for the paper's social/core hierarchy.
    """
    for node in nodes.values():
        node.dcsoc_role = "leaf"
        node.dcsoc_parent = None
        node.dcsoc_children = []
        node.dcsoc_core_neighbors = []

    edges: list[tuple[int, int]] = []
    cluster_ids = sorted(cluster_mgr.cluster_to_head)
    for index, cluster_id in enumerate(cluster_ids):
        core_id = cluster_mgr.cluster_to_head[cluster_id]
        core = nodes[core_id]
        core.dcsoc_role = "core"
        for member_id in sorted(cluster_mgr.cluster_to_members[cluster_id]):
            if member_id == core_id or not nodes[member_id].is_active:
                continue
            nodes[member_id].dcsoc_parent = core_id
            core.dcsoc_children.append(member_id)
            edges.append((core_id, member_id))
        if index:
            parent_core = cluster_mgr.cluster_to_head[cluster_ids[index - 1]]
            core.dcsoc_parent = parent_core
            nodes[parent_core].dcsoc_children.append(core_id)
            nodes[parent_core].dcsoc_core_neighbors.append(core_id)
            core.dcsoc_core_neighbors.append(parent_core)
            edges.append((parent_core, core_id))
    cluster_mgr.structural_edges = list(dict.fromkeys(edges))


def get_dcsoc_master(nodes: Dict[int, Node]) -> int:
    """Return the unique active root core of the DC-SoC structural DAG."""
    roots = sorted(
        node.node_id
        for node in nodes.values()
        if (
            node.is_active
            and node.dcsoc_role == "core"
            and node.dcsoc_parent is None
        )
    )
    if len(roots) != 1:
        raise ValueError(
            "DC-SoC structure must have exactly one active Master/root; "
            f"found {roots}"
        )
    return roots[0]


def repair_dcsoc_after_failure(
    nodes: Dict[int, Node], cluster_mgr: ClusterManager, failed_id: int, was_core: bool
) -> int | None:
    """Perform a local role/relationship transfer; unrelated edges are kept."""
    failed = nodes[failed_id]
    failed.dcsoc_lifecycle = "inactive"
    old_edges = set(cluster_mgr.structural_edges)
    affected = {(a, b) for a, b in old_edges if failed_id in (a, b)}
    cluster_mgr.structural_edges = [edge for edge in cluster_mgr.structural_edges if edge not in affected]
    parent = failed.dcsoc_parent
    children = list(failed.dcsoc_children)
    failed.dcsoc_parent = None
    failed.dcsoc_children = []
    replacement = None
    if was_core and failed.cluster_id is not None:
        eligible = [
            nid for nid in cluster_mgr.cluster_to_members.get(failed.cluster_id, [])
            if nid != failed_id and nodes[nid].is_active
        ]
        if eligible:
            replacement = max(eligible, key=lambda nid: (len(nodes[nid].original_neighbors), -nid))
            repl = nodes[replacement]
            repl.dcsoc_role = "core"
            repl.is_cluster_head = True
            repl.dcsoc_parent = parent if parent != replacement else None
            inherited = [cid for cid in children if cid != replacement and nodes[cid].is_active]
            repl.dcsoc_children = list(dict.fromkeys(repl.dcsoc_children + inherited))
            if parent is not None and nodes[parent].is_active and parent != replacement:
                nodes[parent].dcsoc_children = [replacement if x == failed_id else x for x in nodes[parent].dcsoc_children]
                cluster_mgr.structural_edges.append((parent, replacement))
            for child_id in inherited:
                nodes[child_id].dcsoc_parent = replacement
                cluster_mgr.structural_edges.append((replacement, child_id))
            cluster_mgr.cluster_to_head[failed.cluster_id] = replacement
            cluster_mgr.core_replacement_count += 1
        failed.is_cluster_head = False
        failed.dcsoc_role = "leaf"
    cluster_mgr.structural_edges = list(dict.fromkeys(cluster_mgr.structural_edges))
    changed = len(old_edges.symmetric_difference(set(cluster_mgr.structural_edges)))
    cluster_mgr.structural_repair_count += 1
    cluster_mgr.topology_edges_changed += changed
    cluster_mgr.repair_control_events += max(1, len(affected))
    return replacement


def reinstate_dcsoc_as_leaf(nodes: Dict[int, Node], cluster_mgr: ClusterManager, node_id: int) -> None:
    node = nodes[node_id]
    node.dcsoc_lifecycle = "returned"
    node.dcsoc_role = "leaf"
    node.is_cluster_head = False
    core_id = cluster_mgr.cluster_to_head.get(node.cluster_id)
    node.dcsoc_parent = core_id if core_id != node_id else None
    node.dcsoc_children = []
    if core_id is not None and core_id != node_id:
        nodes[core_id].dcsoc_children = list(dict.fromkeys(nodes[core_id].dcsoc_children + [node_id]))
        edge = (core_id, node_id)
        if edge not in cluster_mgr.structural_edges:
            cluster_mgr.structural_edges.append(edge)
            cluster_mgr.topology_edges_changed += 1


def recluster_dcsoc(nodes: Dict[int, Node], cluster_mgr: ClusterManager, eps: float, min_samples: int) -> ClusterManager:
    """Regenerate from current online physical state at an explicit du boundary."""
    active = {nid: node for nid, node in nodes.items() if node.is_active}
    before = set(cluster_mgr.structural_edges)
    new = assign_dcsoc_clusters(active, eps=eps, min_samples=min_samples)
    cluster_mgr.cluster_to_members = new.cluster_to_members
    cluster_mgr.cluster_to_head = new.cluster_to_head
    cluster_mgr.structural_edges = new.structural_edges
    cluster_mgr.structural_generation += 1
    cluster_mgr.recluster_count += 1
    cluster_mgr.topology_edges_changed += len(before.symmetric_difference(set(new.structural_edges)))
    return cluster_mgr
    
def assign_dcsoc_clusters(
        nodes: Dict[int, Node],
        eps: float,
        min_samples: int,
        ) -> ClusterManager:
    """
    Build the DC-SoC-inspired density-clustered dissemination overlay.

    DBSCAN is applied to the all-pairs shortest-path hop-distance
    matrix of the original physical overlay.

    This construction is separate from the canonical AHBN controller.
    """

    if eps <= 0:
        raise ValueError("DC-SoC eps must be > 0")

    if min_samples <= 0:
        raise ValueError("DC-SoC min_samples must be > 0")

    if not nodes:
        return ClusterManager()

    node_ids = sorted(nodes.keys())

    # Reset existing cluster-overlay state.
    for node in nodes.values():
        node.cluster_id = None
        node.is_cluster_head = False
        node.gateway_neighbors = []
        node.dcsoc_parent = None
        node.dcsoc_children = []
        node.dcsoc_core_neighbors = []

    # ------------------------------------------------------------
    # Reconstruct the original physical graph.
    # ------------------------------------------------------------

    graph = nx.Graph()
    graph.add_nodes_from(node_ids)

    for node_id in node_ids:
        for nbr_id in nodes[node_id].original_neighbors:
            if nbr_id in nodes:
                graph.add_edge(node_id, nbr_id)

    # ------------------------------------------------------------
    # Build all-pairs shortest-path distance matrix.
    # ------------------------------------------------------------

    index_of = {
        node_id: idx
        for idx, node_id in enumerate(node_ids)
    }

    n = len(node_ids)

    unreachable = float(n + 1)

    distance_matrix = np.full(
        (n, n),
        unreachable,
        dtype=float,
    )

    np.fill_diagonal(
        distance_matrix,
        0.0,
    )

    for source_id, distances in nx.all_pairs_shortest_path_length(graph):

        i = index_of[source_id]

        for target_id, distance in distances.items():

            j = index_of[target_id]

            distance_matrix[i, j] = float(distance)

    # ------------------------------------------------------------
    # DBSCAN clustering.
    # ------------------------------------------------------------

    labels = DBSCAN(
        eps=float(eps),
        min_samples=int(min_samples),
        metric="precomputed",
    ).fit_predict(distance_matrix)

    raw_assignments = {
        node_id: int(label)
        for node_id, label in zip(node_ids, labels)
    }

    established_labels = sorted(
        {
            label
            for label in raw_assignments.values()
            if label >= 0
        }
    )

    # ------------------------------------------------------------
    # If DBSCAN finds no cluster, use one cluster containing all
    # nodes so the baseline remains executable.
    # ------------------------------------------------------------

    if not established_labels:

        assignments = {
            node_id: 0
            for node_id in node_ids
        }

    else:

        label_members = {
            label: []
            for label in established_labels
        }

        for node_id, label in raw_assignments.items():

            if label >= 0:
                label_members[label].append(node_id)

        assignments = dict(raw_assignments)

        # --------------------------------------------------------
        # Attach DBSCAN noise nodes (-1) to their nearest cluster.
        # --------------------------------------------------------

        for node_id, label in raw_assignments.items():

            if label >= 0:
                continue

            i = index_of[node_id]

            best_label = min(
                established_labels,
                key=lambda candidate_label: (
                    min(
                        distance_matrix[
                            i,
                            index_of[member_id],
                        ]
                        for member_id in label_members[candidate_label]
                    ),
                    candidate_label,
                ),
            )

            assignments[node_id] = best_label
            label_members[best_label].append(node_id)

        # Normalize cluster IDs to 0, 1, 2, ...
        remap = {
            old_label: new_label
            for new_label, old_label in enumerate(
                sorted(set(assignments.values()))
            )
        }

        assignments = {
            node_id: remap[label]
            for node_id, label in assignments.items()
        }

    # ------------------------------------------------------------
    # Populate existing ClusterManager.
    # ------------------------------------------------------------

    cluster_mgr = ClusterManager(
        head_selection="highest_physical_degree"
    )

    for node_id in node_ids:

        cluster_id = assignments[node_id]

        nodes[node_id].cluster_id = cluster_id

        cluster_mgr.cluster_to_members.setdefault(
            cluster_id,
            [],
        ).append(node_id)

    # ------------------------------------------------------------
    # Choose one head per cluster:
    #
    # highest physical degree,
    # tie -> lowest node ID.
    # ------------------------------------------------------------

    for cluster_id, members in cluster_mgr.cluster_to_members.items():

        members.sort()

        head_id = max(
            members,
            key=lambda nid: (
                len(nodes[nid].original_neighbors),
                -nid,
            ),
        )

        cluster_mgr.cluster_to_head[cluster_id] = head_id

        nodes[head_id].is_cluster_head = True

    # ------------------------------------------------------------
    # Create logical inter-cluster head chain, matching the
    # existing Structured overlay representation.
    # ------------------------------------------------------------

    cluster_ids = sorted(
        cluster_mgr.cluster_to_head.keys()
    )

    for i in range(len(cluster_ids) - 1):

        left = cluster_mgr.cluster_to_head[
            cluster_ids[i]
        ]

        right = cluster_mgr.cluster_to_head[
            cluster_ids[i + 1]
        ]

        nodes[left].gateway_neighbors.append(right)
        nodes[right].gateway_neighbors.append(left)

    cluster_mgr.initial_clustering_count = 1
    _build_dcsoc_structure(nodes, cluster_mgr)
    return cluster_mgr
