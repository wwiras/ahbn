"""
Stage 3.4 — DC-SoC Sanity Validation
S4: Cluster-head relay behaviour correct

Runs the production DC-SoC strategy and simulator while observing the
strategy calls and transport events they genuinely produce. The independent
oracle derives cluster heads and valid relay targets from topology/node state;
it does not call the production cluster-head selection logic.
"""

from collections import defaultdict

import networkx as nx

from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED = 42
N = 30
BA_M = 3
EPS = 2.0
MIN_SAMPLES = 3

# Frozen values used by S3 and configs/stage3_dcsoc.yaml.
FANOUT = 3
INTER_FANOUT = 1
BASE_DELAY = 1.0
JITTER = 0.2
MESSAGE_ID = "dcsoc-s4-transaction"


class ObservingDCSOCStrategy(DCSOCStrategy):
    """Record production target selections without altering them."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.selection_events = []

    def select_targets(self, node, message, simulator):
        targets = super().select_targets(node, message, simulator)
        self.selection_events.append(
            {
                "node_id": node.node_id,
                "is_cluster_head": node.is_cluster_head,
                "message_id": message.message_id,
                "targets": list(targets),
            }
        )
        return targets


class ObservingSimulator(Simulator):
    """Record genuine non-self transport events without changing them."""

    def __init__(self, *args, **kwargs):
        self.forwarding_events = []
        super().__init__(*args, **kwargs)

    def schedule_event(self, time, priority, event_type, payload):
        if event_type == "receive" and payload["src_id"] != payload["dst_id"]:
            self.forwarding_events.append(
                (payload["src_id"], payload["dst_id"], payload["message"].message_id)
            )
        super().schedule_event(time, priority, event_type, payload)


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S4 — Cluster-head relay behaviour correct")
    print("=" * 72)

    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    cluster_manager = assign_dcsoc_clusters(
        nodes,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )

    memberships = defaultdict(list)
    for node_id, node in nodes.items():
        memberships[node.cluster_id].append(node_id)
    for members in memberships.values():
        members.sort()

    non_noise_clusters = sorted(
        cluster_id for cluster_id in memberships if cluster_id != -1
    )
    assert non_noise_clusters, "FAIL: no non-noise cluster is available for S4."

    # Independent oracle: use physical degree and node-ID tie-breaking directly.
    expected_heads = {
        cluster_id: min(
            memberships[cluster_id],
            key=lambda node_id: (-graph.degree[node_id], node_id),
        )
        for cluster_id in non_noise_clusters
    }
    noise_nodes = set(memberships.get(-1, []))
    assert not (noise_nodes & set(expected_heads.values())), (
        "FAIL: independent oracle selected a noise node as a cluster head."
    )

    # Select the first cluster with a non-head member, then its lowest-ID member.
    source_cluster = next(
        (
            cluster_id
            for cluster_id in non_noise_clusters
            if any(
                node_id != expected_heads[cluster_id]
                for node_id in memberships[cluster_id]
            )
        ),
        None,
    )
    assert source_cluster is not None, "FAIL: no deterministic non-CH source exists."
    expected_ch = expected_heads[source_cluster]
    source_node = min(
        node_id
        for node_id in memberships[source_cluster]
        if node_id != expected_ch
    )
    actual_ch = cluster_manager.cluster_to_head.get(source_cluster)

    strategy = ObservingDCSOCStrategy(
        fanout=FANOUT,
        inter_fanout=INTER_FANOUT,
    )
    simulator = ObservingSimulator(
        nodes=nodes,
        strategy=strategy,
        seed=SEED,
        base_delay=BASE_DELAY,
        jitter=JITTER,
        cluster_manager=cluster_manager,
        experiment_name="stage3_dcsoc_s4",
        strategy_name="dcsoc",
    )

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Topology nodes      : {graph.number_of_nodes()}")
    print(f"  Topology edges      : {graph.number_of_edges()}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nCluster summary:")
    for cluster_id in sorted(memberships):
        print(f"  Cluster {cluster_id}:")
        print(f"    Members           : {memberships[cluster_id]}")
        if cluster_id == -1:
            print("    Expected CH       : None (noise excluded)")
            print("    Actual CH         : None (noise excluded)")
        else:
            print(f"    Expected CH       : {expected_heads[cluster_id]}")
            print(f"    Actual CH         : {cluster_manager.cluster_to_head.get(cluster_id)}")

    print("\nTransaction:")
    print(f"  Source node         : {source_node}")
    print(f"  Source selection    : lowest-ID non-CH member of Cluster {source_cluster}")
    print(f"  Source cluster      : {source_cluster}")
    print(f"  Cluster head        : {expected_ch}")

    assert actual_ch == expected_ch, (
        f"FAIL: independently expected CH {expected_ch}, actual CH {actual_ch}."
    )
    assert not nodes[source_node].is_cluster_head, (
        f"FAIL: deterministic source {source_node} is flagged as a cluster head."
    )

    simulator.inject_message(source_node, MESSAGE_ID)
    simulator.run()

    ch_events = [
        event
        for event in strategy.selection_events
        if event["node_id"] == expected_ch and event["message_id"] == MESSAGE_ID
    ]
    ch_reached = MESSAGE_ID in nodes[expected_ch].seen_messages
    ch_path_exercised = len(ch_events) == 1
    ch_targets = ch_events[0]["targets"] if ch_path_exercised else []

    # Independently derive the only valid structured targets: active heads of
    # other non-noise clusters that occur on this head's gateway overlay.
    other_clusters = [
        cluster_id for cluster_id in non_noise_clusters if cluster_id != source_cluster
    ]
    other_expected_heads = {expected_heads[cid] for cid in other_clusters}
    valid_relay_targets = sorted(
        gateway_id
        for gateway_id in nodes[expected_ch].gateway_neighbors
        if gateway_id in other_expected_heads
        and gateway_id in nodes
        and nodes[gateway_id].is_active
        and nodes[gateway_id].cluster_id != -1
    )

    local_candidates = {
        neighbor_id
        for neighbor_id in nodes[expected_ch].neighbors
        if neighbor_id != expected_ch
        and neighbor_id in nodes
        and nodes[neighbor_id].is_active
        and nodes[neighbor_id].cluster_id == source_cluster
    }
    valid_targets = local_candidates | set(valid_relay_targets)
    invalid_targets = sorted(target for target in ch_targets if target not in valid_targets)
    observed_relay_targets = sorted(set(ch_targets) & set(valid_relay_targets))
    observed_local_targets = sorted(set(ch_targets) & local_candidates)

    expected_gateway_count = min(INTER_FANOUT, FANOUT, len(valid_relay_targets))
    expected_local_count = min(
        FANOUT - expected_gateway_count,
        len(local_candidates),
    )
    budget_ok = (
        len(ch_targets) == expected_gateway_count + expected_local_count
        and len(ch_targets) == len(set(ch_targets))
        and len(ch_targets) <= FANOUT
    )
    relay_selection_ok = len(observed_relay_targets) == expected_gateway_count
    local_selection_ok = len(observed_local_targets) == expected_local_count
    outbound_events = [
        receiver
        for sender, receiver, message_id in simulator.forwarding_events
        if sender == expected_ch and message_id == MESSAGE_ID
    ]
    transport_ok = sorted(outbound_events) == sorted(ch_targets)
    no_noise_target = all(target not in noise_nodes for target in ch_targets)
    no_fabricated_relay = all(
        nodes[target].cluster_id == source_cluster
        for target in ch_targets
    ) if not other_clusters else True

    ch_relay_ok = all(
        [
            ch_path_exercised,
            budget_ok,
            relay_selection_ok,
            local_selection_ok,
            transport_ok,
            not invalid_targets,
            no_noise_target,
            no_fabricated_relay,
        ]
    )

    print("\nCH relay validation:")
    print(f"  CH reached          : {pass_fail(ch_reached)}")
    print(f"  CH relay path       : {pass_fail(ch_relay_ok)}")
    print(f"  Other clusters      : {len(other_clusters)}")
    print(f"  Valid relay targets : {valid_relay_targets}")
    print(f"  Selected relay      : {observed_relay_targets}")
    print(f"  Selected local      : {observed_local_targets}")
    print(f"  Invalid targets     : {len(invalid_targets)}")
    print(f"  Fanout budget       : {len(ch_targets)}/{FANOUT}")

    assert ch_reached, f"FAIL: transaction did not reach expected CH {expected_ch}."
    assert ch_path_exercised, (
        f"FAIL: expected exactly one first-receive CH strategy call; observed {len(ch_events)}."
    )
    assert not invalid_targets, f"FAIL: CH produced invalid targets: {invalid_targets}."
    assert no_noise_target, "FAIL: CH produced a noise-node relay target."
    assert no_fabricated_relay, "FAIL: CH fabricated an inter-cluster relay target."
    assert budget_ok, f"FAIL: CH target budget/output is incorrect: {ch_targets}."
    assert relay_selection_ok, (
        "FAIL: CH structured relay selection contradicts independently derived targets; "
        f"valid={valid_relay_targets}, observed={observed_relay_targets}."
    )
    assert local_selection_ok, (
        "FAIL: CH local remainder selection contradicts the frozen budget semantics."
    )
    assert transport_ok, (
        f"FAIL: CH selections {ch_targets} do not match scheduled transport {outbound_events}."
    )

    if not other_clusters:
        print("\nInter-cluster dissemination : NOT EXERCISED")
    else:
        print(
            "\nInter-cluster dissemination : "
            + ("OBSERVED" if observed_relay_targets else "NOT OBSERVED")
        )

    print("\nS4 result:")
    print("  Cluster-head relay behaviour : PASS")
    print("\n" + "=" * 72)
    print("S4 RESULT: PASS")
    print("=" * 72)


if __name__ == "__main__":
    main()
