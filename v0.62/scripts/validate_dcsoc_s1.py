"""
Stage 3.4 — DC-SoC Sanity Validation
S1: Cluster assignment correctness

Purpose:
    Verify that the frozen Stage 3.3 DC-SoC implementation assigns
    every topology node consistently to exactly one density cluster.

Important:
    - Uses the same Node construction path as the real simulator.
    - Uses the frozen Stage 3.3 DC-SoC parameters.
    - Does NOT evaluate cluster-head correctness.
    - Does NOT evaluate dissemination or performance.
"""

from collections import Counter, defaultdict

import networkx as nx

from ahbn.topology import (
    assign_dcsoc_clusters,
    build_nodes_from_graph,
)


# =================================================================
# Stage 3.4 deterministic sanity-test topology
# =================================================================

SEED = 42
N = 30
BA_M = 3


# =================================================================
# Frozen Stage 3.3 DC-SoC parameters
#
# Must match:
#
#     configs/stage3_dcsoc.yaml
#
# dcsoc:
#     eps: 2.0
#     min_samples: 3
#
# These values are NOT tuned during Stage 3.4.
# =================================================================

EPS = 2.0
MIN_SAMPLES = 3


def main():

    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S1 — Cluster assignment correct")
    print("=" * 72)

    # -------------------------------------------------------------
    # 1. Build a deterministic physical topology
    # -------------------------------------------------------------

    graph = nx.barabasi_albert_graph(
        n=N,
        m=BA_M,
        seed=SEED,
    )

    topology_nodes = set(
        graph.nodes()
    )

    print("\nTest configuration:")
    print(f"  Topology type      : BA")
    print(f"  Topology nodes     : {len(topology_nodes)}")
    print(f"  Topology edges     : {graph.number_of_edges()}")
    print(f"  BA m               : {BA_M}")
    print(f"  Seed               : {SEED}")
    print(f"  DBSCAN eps         : {EPS}")
    print(f"  DBSCAN min_samples : {MIN_SAMPLES}")

    # -------------------------------------------------------------
    # 2. Construct Nodes exactly as the simulator does
    # -------------------------------------------------------------
    #
    # v0.6(10):
    #
    # build_nodes_from_graph(graph)
    #
    # creates:
    #
    # Dict[int, Node]
    #
    # and Node.__post_init__ preserves the physical neighbour
    # set in original_neighbors.
    # -------------------------------------------------------------

    nodes = build_nodes_from_graph(
        graph
    )

    assert isinstance(nodes, dict), (
        "FAIL: build_nodes_from_graph() did not return a dictionary."
    )

    assert set(nodes.keys()) == topology_nodes, (
        "FAIL: Simulator node dictionary does not match topology nodes."
    )

    print("\nNode construction:")
    print(f"  Nodes constructed  : {len(nodes)}")
    print("  Node dictionary    : PASS")

    # -------------------------------------------------------------
    # 3. Verify physical topology was preserved before clustering
    # -------------------------------------------------------------

    physical_topology_errors = []

    for node_id in sorted(topology_nodes):

        expected_neighbors = set(
            graph.neighbors(node_id)
        )

        actual_neighbors = set(
            nodes[node_id].original_neighbors
        )

        if expected_neighbors != actual_neighbors:

            physical_topology_errors.append(
                (
                    node_id,
                    sorted(expected_neighbors),
                    sorted(actual_neighbors),
                )
            )

    assert not physical_topology_errors, (
        "FAIL: original_neighbors does not reproduce the physical "
        f"topology: {physical_topology_errors}"
    )

    print("  Physical overlay   : PASS")

    # -------------------------------------------------------------
    # 4. Run the real Stage 3.3 DC-SoC cluster construction
    # -------------------------------------------------------------

    cluster_manager = assign_dcsoc_clusters(
        nodes,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )

    assert cluster_manager is not None, (
        "FAIL: assign_dcsoc_clusters() returned None."
    )

    # -------------------------------------------------------------
    # 5. Read assignments from the real Node state
    # -------------------------------------------------------------
    #
    # assign_dcsoc_clusters() writes:
    #
    #     nodes[node_id].cluster_id = cluster_id
    #
    # It does NOT return an assignment dictionary.
    # -------------------------------------------------------------

    assignments = {
        node_id: node.cluster_id
        for node_id, node in nodes.items()
    }

    print("\nNode -> cluster assignments:")

    for node_id in sorted(assignments):

        print(
            f"  Node {node_id:2d} -> "
            f"Cluster {assignments[node_id]}"
        )

    # -------------------------------------------------------------
    # 6. S1 invariant: every node must have a cluster
    # -------------------------------------------------------------

    unassigned_nodes = [
        node_id
        for node_id, cluster_id in assignments.items()
        if cluster_id is None
    ]

    assert not unassigned_nodes, (
        "FAIL: Nodes without a DC-SoC cluster assignment: "
        f"{unassigned_nodes}"
    )

    # -------------------------------------------------------------
    # 7. Validate cluster IDs
    # -------------------------------------------------------------
    #
    # Important:
    #
    # v0.6(10) attaches DBSCAN noise nodes to the nearest established
    # cluster. Therefore final node.cluster_id values should be
    # normalized non-negative integers.
    #
    # There should be NO -1 final cluster ID.
    # -------------------------------------------------------------

    invalid_cluster_ids = {
        node_id: cluster_id
        for node_id, cluster_id in assignments.items()
        if (
            not isinstance(cluster_id, int)
            or cluster_id < 0
        )
    }

    assert not invalid_cluster_ids, (
        "FAIL: Invalid final cluster IDs found: "
        f"{invalid_cluster_ids}"
    )

    # -------------------------------------------------------------
    # 8. Build memberships independently from Node state
    # -------------------------------------------------------------

    reconstructed_clusters = defaultdict(list)

    for node_id, cluster_id in assignments.items():

        reconstructed_clusters[
            cluster_id
        ].append(
            node_id
        )

    for members in reconstructed_clusters.values():
        members.sort()

    print("\nCluster membership:")

    for cluster_id in sorted(reconstructed_clusters):

        members = reconstructed_clusters[
            cluster_id
        ]

        print(
            f"  Cluster {cluster_id}: "
            f"{members} "
            f"[n={len(members)}]"
        )

    # -------------------------------------------------------------
    # 9. Validate ClusterManager coverage
    # -------------------------------------------------------------

    manager_clusters = (
        cluster_manager.cluster_to_members
    )

    manager_nodes = set()

    duplicate_manager_membership = []

    membership_counter = Counter()

    for cluster_id, members in manager_clusters.items():

        for node_id in members:

            membership_counter[
                node_id
            ] += 1

            manager_nodes.add(
                node_id
            )

    duplicate_manager_membership = [
        node_id
        for node_id, count in membership_counter.items()
        if count != 1
    ]

    missing_from_manager = (
        topology_nodes
        - manager_nodes
    )

    unknown_in_manager = (
        manager_nodes
        - topology_nodes
    )

    assert not missing_from_manager, (
        "FAIL: Nodes missing from ClusterManager: "
        f"{sorted(missing_from_manager)}"
    )

    assert not unknown_in_manager, (
        "FAIL: Unknown nodes present in ClusterManager: "
        f"{sorted(unknown_in_manager)}"
    )

    assert not duplicate_manager_membership, (
        "FAIL: Nodes appear in more than one ClusterManager cluster: "
        f"{sorted(duplicate_manager_membership)}"
    )

    # -------------------------------------------------------------
    # 10. Cross-check Node state against ClusterManager
    # -------------------------------------------------------------

    inconsistencies = []

    for cluster_id, members in manager_clusters.items():

        for node_id in members:

            node_cluster = (
                nodes[node_id].cluster_id
            )

            if node_cluster != cluster_id:

                inconsistencies.append(
                    (
                        node_id,
                        node_cluster,
                        cluster_id,
                    )
                )

    assert not inconsistencies, (
        "FAIL: Node.cluster_id disagrees with ClusterManager: "
        f"{inconsistencies}"
    )

    # -------------------------------------------------------------
    # 11. Cluster-ID normalization check
    # -------------------------------------------------------------
    #
    # v0.6(10) explicitly remaps IDs to:
    #
    #     0, 1, 2, ...
    #
    # Therefore test that final IDs are contiguous.
    # -------------------------------------------------------------

    actual_cluster_ids = sorted(
        reconstructed_clusters.keys()
    )

    expected_cluster_ids = list(
        range(
            len(actual_cluster_ids)
        )
    )

    assert actual_cluster_ids == expected_cluster_ids, (
        "FAIL: Cluster IDs are not normalized/contiguous: "
        f"actual={actual_cluster_ids}, "
        f"expected={expected_cluster_ids}"
    )

    # -------------------------------------------------------------
    # 12. S1 summary
    # -------------------------------------------------------------

    print("\nS1 invariant summary:")

    print(
        f"  Topology nodes             : "
        f"{len(topology_nodes)}"
    )

    print(
        f"  Node objects               : "
        f"{len(nodes)}"
    )

    print(
        f"  Assigned nodes             : "
        f"{len(assignments) - len(unassigned_nodes)}"
    )

    print(
        f"  Unassigned nodes           : "
        f"{len(unassigned_nodes)}"
    )

    print(
        f"  ClusterManager nodes       : "
        f"{len(manager_nodes)}"
    )

    print(
        f"  Missing manager nodes      : "
        f"{len(missing_from_manager)}"
    )

    print(
        f"  Unknown manager nodes      : "
        f"{len(unknown_in_manager)}"
    )

    print(
        f"  Duplicate memberships      : "
        f"{len(duplicate_manager_membership)}"
    )

    print(
        f"  Node/manager inconsistencies: "
        f"{len(inconsistencies)}"
    )

    print(
        f"  Final clusters             : "
        f"{len(actual_cluster_ids)}"
    )

    # -------------------------------------------------------------
    # 13. PASS
    # -------------------------------------------------------------

    print("\n" + "=" * 72)

    print(
        "S1 PASS — Every node has exactly one valid and "
        "internally consistent DC-SoC cluster assignment."
    )

    print("=" * 72)


if __name__ == "__main__":
    main()