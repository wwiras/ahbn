"""
Stage 3.4 — DC-SoC Sanity Validation
S2: Cluster-head identification correctness

Independently verifies the frozen Stage 3.3 DC-SoC cluster-head rule.
This validator does not exercise dissemination.
"""

from collections import defaultdict

import networkx as nx

from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED = 42
N = 30
BA_M = 3
EPS = 2.0
MIN_SAMPLES = 3


def build_dcsoc():
    """Construct the deterministic S1 topology through the real DC-SoC path."""
    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    cluster_manager = assign_dcsoc_clusters(
        nodes,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )
    return graph, nodes, cluster_manager


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S2 — Cluster-head identification correct")
    print("=" * 72)

    graph, nodes, cluster_manager = build_dcsoc()

    print("\nTest configuration:")
    print("  Topology type      : BA")
    print(f"  Topology nodes     : {graph.number_of_nodes()}")
    print(f"  Topology edges     : {graph.number_of_edges()}")
    print(f"  BA m               : {BA_M}")
    print(f"  Seed               : {SEED}")
    print(f"  DBSCAN eps         : {EPS}")
    print(f"  DBSCAN min_samples : {MIN_SAMPLES}")

    # S1 maps Node -> DBSCAN cluster. S2 independently reverses that
    # final Node state to obtain Cluster -> members, then validates heads.
    memberships = defaultdict(list)
    for node_id, node in nodes.items():
        memberships[node.cluster_id].append(node_id)
    for members in memberships.values():
        members.sort()

    non_noise_clusters = sorted(
        cluster_id for cluster_id in memberships if cluster_id != -1
    )
    noise_nodes = sorted(memberships.get(-1, []))
    node_flag_heads = sorted(
        node_id for node_id, node in nodes.items() if node.is_cluster_head
    )

    failures = []
    membership_checks = []
    selection_checks = []
    tie_checks = []

    print("\nCluster-head validation:")

    for cluster_id in non_noise_clusters:
        members = memberships[cluster_id]
        # Independent oracle: degree comes directly from the original
        # NetworkX physical overlay, not a DC-SoC selection helper.
        member_degrees = {node_id: graph.degree[node_id] for node_id in members}
        maximum_degree = max(member_degrees.values()) if members else None
        tied_candidates = sorted(
            node_id
            for node_id, degree in member_degrees.items()
            if degree == maximum_degree
        )
        expected_ch = tied_candidates[0] if tied_candidates else None
        actual_ch = cluster_manager.cluster_to_head.get(cluster_id)
        flagged_in_cluster = [node_id for node_id in members if nodes[node_id].is_cluster_head]

        nonempty_ok = bool(members)
        exactly_one_ok = len(flagged_in_cluster) == 1 and actual_ch in flagged_in_cluster
        membership_ok = actual_ch in members
        selection_ok = actual_ch == expected_ch
        tie_observed = len(tied_candidates) > 1
        tie_ok = not tie_observed or actual_ch == min(tied_candidates)

        membership_checks.append(membership_ok)
        selection_checks.append(selection_ok)
        if tie_observed:
            tie_checks.append(tie_ok)

        print(f"\n  Cluster {cluster_id}:")
        print(f"    Members          : {members}")
        print(f"    Member degrees   : {[(nid, member_degrees[nid]) for nid in members]}")
        print(f"    Maximum degree   : {maximum_degree}")
        if tie_observed:
            print(f"    Tied candidates  : {tied_candidates}")
        print(f"    Expected CH      : Node {expected_ch}")
        print(f"    Actual CH        : Node {actual_ch}")
        print(f"    Membership check : {'PASS' if membership_ok else 'FAIL'}")
        print(f"    Selection check  : {'PASS' if selection_ok else 'FAIL'}")

        if not nonempty_ok:
            failures.append(f"Cluster {cluster_id}: non-noise cluster is empty")
        if not exactly_one_ok:
            failures.append(
                f"Cluster {cluster_id}: expected exactly one consistent CH; "
                f"manager={actual_ch}, node flags={flagged_in_cluster}"
            )
        if not membership_ok:
            failures.append(
                f"Cluster {cluster_id}: actual CH {actual_ch} is not a member; "
                f"members={members}"
            )
        if not selection_ok:
            failures.append(
                f"Cluster {cluster_id}: expected CH {expected_ch}, actual CH {actual_ch}; "
                f"member degrees={member_degrees}"
            )
        if not tie_ok:
            failures.append(
                f"Cluster {cluster_id}: tie-break violation; tied candidates="
                f"{tied_candidates}, expected CH {expected_ch}, actual CH {actual_ch}"
            )

    manager_cluster_ids = set(cluster_manager.cluster_to_head)
    expected_cluster_ids = set(non_noise_clusters)
    one_per_cluster_ok = (
        manager_cluster_ids == expected_cluster_ids
        and len(node_flag_heads) == len(non_noise_clusters)
        and all(
            sum(nodes[node_id].is_cluster_head for node_id in memberships[cluster_id]) == 1
            for cluster_id in non_noise_clusters
        )
    )
    noise_excluded_ok = all(not nodes[node_id].is_cluster_head for node_id in noise_nodes)

    _, replay_nodes, replay_manager = build_dcsoc()
    first_assignments = {
        cluster_id: cluster_manager.cluster_to_head[cluster_id]
        for cluster_id in sorted(cluster_manager.cluster_to_head)
    }
    replay_assignments = {
        cluster_id: replay_manager.cluster_to_head[cluster_id]
        for cluster_id in sorted(replay_manager.cluster_to_head)
    }
    same_seed_ok = (
        first_assignments == replay_assignments
        and {nid: node.cluster_id for nid, node in nodes.items()}
        == {nid: node.cluster_id for nid, node in replay_nodes.items()}
    )

    if not one_per_cluster_ok:
        failures.append(
            "Global: not exactly one CH per non-noise cluster; "
            f"manager clusters={sorted(manager_cluster_ids)}, "
            f"expected clusters={non_noise_clusters}, flagged heads={node_flag_heads}"
        )
    if not noise_excluded_ok:
        failures.append(f"Global: noise nodes selected as CHs: {sorted(set(noise_nodes) & set(node_flag_heads))}")
    if not same_seed_ok:
        failures.append(
            "Global: same-seed reconstruction changed assignments; "
            f"first={first_assignments}, replay={replay_assignments}"
        )

    print("\nGlobal checks:")
    print(f"  Non-noise clusters             : {len(non_noise_clusters)}")
    print(f"  Cluster heads identified       : {len(node_flag_heads)}")
    print(f"  One CH per cluster             : {'PASS' if one_per_cluster_ok else 'FAIL'}")
    print(f"  Every CH belongs to cluster    : {'PASS' if all(membership_checks) else 'FAIL'}")
    print(f"  Highest-degree selection       : {'PASS' if all(selection_checks) else 'FAIL'}")
    if tie_checks:
        print(f"  Deterministic tie-breaking     : {'PASS' if all(tie_checks) else 'FAIL'} ({len(tie_checks)} natural tie(s) tested)")
    else:
        print("  Deterministic tie-breaking     : NOT OBSERVED (no natural maximum-degree tie)")
    print(f"  Noise excluded from CHs        : {'PASS' if noise_excluded_ok else 'FAIL'}")
    print(f"  Same-seed reproducibility      : {'PASS' if same_seed_ok else 'FAIL'}")

    print("\n" + "=" * 72)
    if failures:
        print("S2 RESULT: FAIL")
        print("Cluster-head identification is incorrect:")
        for failure in failures:
            print(f"  - {failure}")
        print("=" * 72)
        raise SystemExit(1)

    print("S2 RESULT: PASS")
    print("Cluster-head identification is correct.")
    print("=" * 72)


if __name__ == "__main__":
    main()
