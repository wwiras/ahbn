"""Stage 3.4 S6: validate triggered DC-SoC structural maintenance."""

from collections import defaultdict
import sys
from pathlib import Path

# Allow the validator to be launched directly with the project interpreter.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED, N, BA_M = 42, 30, 3
EPS, MIN_SAMPLES = 2.0, 3
FANOUT, INTER_FANOUT = 3, 1
TRIGGER_TIME = 1.0


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def assignments(nodes, active_only):
    return {
        node_id: node.cluster_id
        for node_id, node in sorted(nodes.items())
        if not active_only or node.is_active
    }


def memberships(node_to_cluster):
    result = defaultdict(list)
    for node_id, cluster_id in node_to_cluster.items():
        result[cluster_id].append(node_id)
    return {cluster_id: sorted(member_ids) for cluster_id, member_ids in sorted(result.items())}


def reconstruct_heads(cluster_members):
    return {
        # Frozen DC-SoC repair uses _select_cluster_head with
        # resource_aware_heads=False, i.e. the lowest active member ID.
        cluster_id: min(member_ids)
        for cluster_id, member_ids in cluster_members.items()
        if cluster_id != -1 and member_ids
    }


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S6 — Structural update works when triggered")
    print("=" * 72)

    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=EPS, min_samples=MIN_SAMPLES)
    simulator = Simulator(
        nodes=nodes,
        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
        seed=SEED,
        cluster_manager=manager,
        experiment_name="stage3_dcsoc_s6",
        strategy_name="dcsoc",
    )

    before_assignments = assignments(nodes, True)
    before_memberships = memberships(before_assignments)
    before_heads = dict(sorted(manager.cluster_to_head.items()))
    before_active = {node_id for node_id, node in nodes.items() if node.is_active}
    before_edges = {
        tuple(sorted((node_id, neighbor_id)))
        for node_id in before_active
        for neighbor_id in nodes[node_id].neighbors
        if neighbor_id in before_active
    }
    initial_repairs = simulator.metrics.cluster_repair_count

    # Leaving the initial CH is the smallest deterministic availability event
    # that guarantees both membership and CH metadata must change. The event is
    # processed by the normal event loop; this validator never calls repair.
    target_node = min(before_heads.values())
    target_cluster = nodes[target_node].cluster_id
    trigger_before = not nodes[target_node].is_active
    physical_degrees = {
        node_id: len(node.original_neighbors) for node_id, node in nodes.items()
    }
    simulator.schedule_event(
        time=TRIGGER_TIME,
        priority=0,
        event_type="churn_leave",
        payload={"node_id": target_node, "churn_rate": 1.0 / N},
    )
    expected_trigger_after = target_node in before_active
    simulator.run(until=TRIGGER_TIME)

    actual_trigger = not nodes[target_node].is_active
    update_executed = simulator.metrics.cluster_repair_count == initial_repairs + 1
    after_assignments = assignments(nodes, True)
    after_memberships = memberships(after_assignments)
    after_heads = dict(sorted(manager.cluster_to_head.items()))
    after_active = {node_id for node_id, node in nodes.items() if node.is_active}
    after_edges = {
        tuple(sorted((node_id, neighbor_id)))
        for node_id in after_active
        for neighbor_id in nodes[node_id].neighbors
        if neighbor_id in after_active
    }

    # Independent oracle for the frozen repair semantics: preserve each active
    # node's density-cluster ID, remove inactive members, and independently
    # reselect heads using physical degree with the lowest-ID tie break.
    expected_assignments = {
        node_id: cluster_id
        for node_id, cluster_id in before_assignments.items()
        if node_id != target_node
    }
    expected_memberships = memberships(expected_assignments)
    expected_heads = reconstruct_heads(expected_memberships)

    cluster_check = expected_assignments == after_assignments
    membership_check = expected_memberships == after_memberships
    ch_check = expected_heads == after_heads
    non_noise = {cluster_id for cluster_id in expected_memberships if cluster_id != -1}
    flagged_heads = {node_id for node_id, node in nodes.items() if node.is_cluster_head}
    noise_nodes = set(expected_memberships.get(-1, []))
    one_cluster_per_node = (
        set(after_assignments) == after_active
        and sum(map(len, after_memberships.values())) == len(after_active)
        and len({node for group in after_memberships.values() for node in group}) == len(after_active)
    )
    one_ch_per_cluster = set(after_heads) == non_noise and len(flagged_heads) == len(non_noise)
    ch_membership = all(head in after_memberships[cid] for cid, head in after_heads.items())
    noise_handling = not (noise_nodes & flagged_heads)
    valid_references = (
        all(node_id in after_active for node_id in after_assignments)
        and all(head in after_active for head in after_heads.values())
        and all(u in after_active and v in after_active for u, v in after_edges)
    )
    structure_changed = (
        before_assignments != after_assignments
        and before_memberships != after_memberships
        and before_heads != after_heads
    )
    forwarding_isolation = (
        isinstance(simulator.strategy, DCSOCStrategy)
        and simulator.controller is None
        and simulator.metrics.mode_switch_count == 0
        and simulator.metrics.fanout_change_count == 0
        and simulator.metrics.adaptation_event_count == 0
    )

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Topology nodes      : {graph.number_of_nodes()}")
    print(f"  Topology edges      : {graph.number_of_edges()}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nFrozen structural-update mechanism:")
    print("  Trigger type        : active node availability transition (churn leave/join)")
    print("  Trigger location    : Simulator.run() -> handle_churn_leave()/handle_churn_join()")
    print("  Update function     : repair_topology_after_churn()")

    print("\nBefore structural change:")
    print(f"  Nodes               : {sorted(before_active)}")
    print(f"  Edges               : {len(before_edges)}")
    print(f"  Clusters            : {before_memberships}")
    print(f"  Cluster heads       : {before_heads}")
    print(f"  Trigger condition   : {'TRUE' if trigger_before else 'FALSE'}")

    print("\nDeterministic trigger event:")
    print("  Change applied      : production churn_leave event")
    print(f"  Affected node/edge  : node {target_node}; {physical_degrees[target_node]} physical edges inactive")
    print(f"  Reason              : node {target_node} is the initial CH of Cluster {target_cluster}")

    print("\nTrigger validation:")
    print(f"  Expected trigger    : {'TRUE' if expected_trigger_after else 'FALSE'}")
    print(f"  Actual trigger      : {'TRUE' if actual_trigger else 'FALSE'}")
    print(f"  Trigger check       : {pass_fail(not trigger_before and expected_trigger_after and actual_trigger)}")
    print(f"  Repair counter      : {initial_repairs} -> {simulator.metrics.cluster_repair_count}")
    print(f"  Update executed     : {pass_fail(update_executed)}")

    print("\nAfter structural update:")
    print(f"  Nodes               : {sorted(after_active)}")
    print(f"  Edges               : {len(after_edges)}")
    print(f"  Clusters            : {after_memberships}")
    print(f"  Cluster heads       : {after_heads}")
    print(f"  Structure changed   : {pass_fail(structure_changed)}")

    print("\nIndependent reconstruction:")
    print(f"  Expected clusters   : {expected_memberships}")
    print(f"  Actual clusters     : {after_memberships}")
    print(f"  Cluster check       : {pass_fail(cluster_check and membership_check)}")
    print(f"  Expected CHs        : {expected_heads}")
    print(f"  Actual CHs          : {after_heads}")
    print(f"  CH check            : {pass_fail(ch_check)}")

    print("\nStructural integrity:")
    print(f"  One cluster/node    : {pass_fail(one_cluster_per_node)}")
    print(f"  One CH/cluster      : {pass_fail(one_ch_per_cluster)}")
    print(f"  CH membership       : {pass_fail(ch_membership)}")
    print(f"  Noise handling      : {pass_fail(noise_handling)}")
    print(f"  Valid references    : {pass_fail(valid_references)}")

    print("\nForwarding-policy isolation:")
    print(f"  Runtime forwarding adaptation introduced : {'NO' if forwarding_isolation else 'YES'}")
    print(f"  Check                                 : {pass_fail(forwarding_isolation)}")

    passed = all((
        not trigger_before, expected_trigger_after, actual_trigger,
        update_executed, structure_changed, cluster_check, membership_check,
        ch_check, one_cluster_per_node, one_ch_per_cluster, ch_membership,
        noise_handling, valid_references, forwarding_isolation,
    ))
    print("\n" + "-" * 72)
    print(f"S6 RESULT: {pass_fail(passed)}")
    print("-" * 72)
    if passed:
        print("The deterministic node-availability change genuinely satisfied the frozen")
        print("DC-SoC structural-update trigger. The resulting active memberships and")
        print("cluster heads matched the independent post-change reconstruction. No")
        print("unrelated runtime forwarding adaptation was introduced.")
    assert passed, "S6 structural-update validation failed."


if __name__ == "__main__":
    main()
