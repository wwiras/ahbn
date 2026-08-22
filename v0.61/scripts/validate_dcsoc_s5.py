"""Stage 3.4 S5: independently sanity-check DC-SoC duplicates."""

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
FANOUT = 3
INTER_FANOUT = 1
BASE_DELAY = 1.0
JITTER = 0.2
MESSAGE_ID = "dcsoc-s5-transaction"


class ObservingSimulator(Simulator):
    """Observe processed receives without changing production semantics."""

    def __init__(self, *args, **kwargs):
        self.reception_events = []
        super().__init__(*args, **kwargs)

    def handle_receive(self, now, dst_id, src_id, message, sent_at=None):
        # Record immediately before/after production handling so this is actual
        # heap/event order, not schedule order (link delays can reorder receives).
        active = self.nodes[dst_id].is_active
        before = self.metrics.messages[message.message_id].duplicate_count
        super().handle_receive(now, dst_id, src_id, message, sent_at)
        after = self.metrics.messages[message.message_id].duplicate_count
        if active:
            self.reception_events.append({
                "time": now, "sender": src_id, "receiver": dst_id,
                "message_id": message.message_id,
                "simulator_increment": after - before,
            })


def role(node_id, source_id, nodes):
    if node_id == source_id:
        return "source"
    if nodes[node_id].is_cluster_head:
        return "CH"
    return "member"


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S5 — Duplicate behaviour plausible")
    print("=" * 72)

    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    cluster_manager = assign_dcsoc_clusters(nodes, eps=EPS, min_samples=MIN_SAMPLES)
    memberships = defaultdict(list)
    for node_id, node in nodes.items():
        memberships[node.cluster_id].append(node_id)
    for members in memberships.values():
        members.sort()

    source_cluster = next(
        cluster_id for cluster_id in sorted(memberships)
        if any(not nodes[node_id].is_cluster_head for node_id in memberships[cluster_id])
    )
    source_id = min(
        node_id for node_id in memberships[source_cluster]
        if not nodes[node_id].is_cluster_head
    )
    simulator = ObservingSimulator(
        nodes=nodes,
        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
        seed=SEED, base_delay=BASE_DELAY, jitter=JITTER,
        cluster_manager=cluster_manager, experiment_name="stage3_dcsoc_s5",
        strategy_name="dcsoc",
    )
    simulator.inject_message(source_id, MESSAGE_ID)
    simulator.run()

    # Independent receiver-based oracle. It uses only the observable processed
    # reception stream and its own set; it calls neither the metric helper being
    # validated nor Node.has_seen(). Source self-reception is the first reception.
    seen = set()
    first_delivery = {}
    duplicates = []
    first_receptions = []
    for index, event in enumerate(simulator.reception_events, start=1):
        receiver = event["receiver"]
        enriched = {**event, "index": index}
        if receiver in seen:
            enriched["first_delivery"] = first_delivery[receiver]
            duplicates.append(enriched)
        else:
            seen.add(receiver)
            first_delivery[receiver] = enriched
            first_receptions.append(enriched)

    record = simulator.metrics.messages[MESSAGE_ID]
    simulator_duplicates = record.duplicate_count
    independent_duplicates = len(duplicates)
    semantics_identified = all(
        event["simulator_increment"] in (0, 1) for event in simulator.reception_events
    )
    first_not_duplicates = all(
        event["simulator_increment"] == 0 for event in first_receptions
    )
    duplicate_increments = all(
        event["simulator_increment"] == 1 for event in duplicates
    )
    accounting_match = independent_duplicates == simulator_duplicates

    for event in duplicates:
        sender, receiver = event["sender"], event["receiver"]
        event["physical_edge"] = graph.has_edge(sender, receiver)
        event["gateway_edge"] = receiver in nodes[sender].gateway_neighbors
        event["valid_policy_edge"] = (
            event["physical_edge"]
            and nodes[sender].cluster_id == nodes[receiver].cluster_id
        ) or event["gateway_edge"]
    structurally_plausible = duplicate_increments and all(
        event["valid_policy_edge"] for event in duplicates
    )

    # Zero is allowed only if the observed transmission structure has no repeated
    # receiver. This baseline naturally has duplicates, so no alternate topology
    # is needed and the zero-only fallback is not applicable.
    repeated_receivers_possible = len(simulator.reception_events) > len(seen)
    zero_justified = independent_duplicates == 0 and not repeated_receivers_possible

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
        print(f"    Cluster head      : {cluster_manager.cluster_to_head.get(cluster_id)}")

    print("\nTransaction:")
    print(f"  Source node         : {source_id}")
    print(f"  Source selection    : lowest-ID non-CH member of Cluster {source_cluster}")

    print("\nDuplicate-accounting semantics:")
    print("  Unit                : processed reception at an active receiver")
    print("  First reception     : marks receiver seen; not a duplicate")
    print("  Later reception     : increments message duplicate count and does not forward")
    print("  Source self-receive : included as the source's first reception")

    print("\nReception / duplicate trace:")
    if not duplicates:
        print("  No duplicate receptions observed.")
    for event in duplicates:
        first = event["first_delivery"]
        sender, receiver = event["sender"], event["receiver"]
        overlay = "physical" if event["physical_edge"] else "gateway"
        print(
            f"  #{event['index']:02d} t={event['time']:.6f}: receiver {receiver} "
            f"first from {first['sender']} (#{first['index']:02d}), duplicate from "
            f"{sender}; cluster={nodes[receiver].cluster_id}; "
            f"roles={role(first['sender'], source_id, nodes)}->"
            f"{role(receiver, source_id, nodes)}, later "
            f"{role(sender, source_id, nodes)}->{role(receiver, source_id, nodes)}; "
            f"overlay={overlay}; edge={'YES' if event['valid_policy_edge'] else 'NO'}"
        )

    print("\nAccounting summary:")
    print(f"  Total transmissions              : {record.total_forwards}")
    print(f"  Unique receivers                 : {len(seen)}")
    print(f"  Independent duplicate count      : {independent_duplicates}")
    print(f"  Simulator-reported duplicates    : {simulator_duplicates}")
    print(f"  Accounting match                 : {pass_fail(accounting_match)}")

    print("\nStructural plausibility:")
    print(f"  Duplicate receptions observed    : {independent_duplicates}")
    print(f"  Valid forwarding/overlay edges   : {pass_fail(structurally_plausible)}")

    print("\nChecks:")
    print(f"  [{pass_fail(semantics_identified)}] Duplicate accounting semantics identified")
    print(f"  [{pass_fail(first_not_duplicates)}] First receptions are not counted as duplicates")
    print(f"  [{pass_fail(structurally_plausible)}] Observed duplicates are structurally plausible")
    print(f"  [{pass_fail(accounting_match)}] Independent duplicate count matches simulator accounting")
    if independent_duplicates:
        print("  [N/A] Zero-duplicate case structurally justified (duplicates observed)")
    else:
        print(f"  [{pass_fail(zero_justified)}] Zero-duplicate case structurally justified")

    passed = all((semantics_identified, first_not_duplicates,
                  structurally_plausible, accounting_match)) and (
        independent_duplicates > 0 or zero_justified
    )
    print("\nFinal result:")
    print(f"  S5 duplicate behaviour plausible: {pass_fail(passed)}")
    assert passed, "S5 duplicate behaviour/accounting validation failed."


if __name__ == "__main__":
    main()
