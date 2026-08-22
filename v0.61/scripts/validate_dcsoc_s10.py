"""Stage 3.4 S10: validate deterministic DC-SoC reproducibility."""

from collections import defaultdict

import networkx as nx

from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED, N, BA_M = 42, 30, 3
EPS, MIN_SAMPLES = 2.0, 3
FANOUT, INTER_FANOUT = 3, 1
BASE_DELAY, JITTER = 1.0, 0.2
SOURCE_ID, TRANSACTION_ID = 0, "1"


class TracingSimulator(Simulator):
    """Observe production forwarding and first receptions without changing them."""

    def __init__(self, *args, **kwargs):
        self.forwarding_events = []
        self.first_receptions = []
        super().__init__(*args, **kwargs)

    def send_message(self, src_id, dst_id, message, now):
        queue_size = len(self.queue)
        super().send_message(src_id, dst_id, message, now)
        if len(self.queue) > queue_size:
            self.forwarding_events.append((src_id, dst_id))

    def handle_receive(self, now, dst_id, src_id, message, sent_at=None):
        first = not self.nodes[dst_id].has_seen(message.message_id)
        super().handle_receive(now, dst_id, src_id, message, sent_at)
        if first and self.nodes[dst_id].has_seen(message.message_id):
            self.first_receptions.append((dst_id, src_id, now))


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def normalized_edges(graph):
    return tuple(sorted((min(left, right), max(left, right)) for left, right in graph.edges()))


def cluster_assignments(nodes):
    return tuple((node_id, node.cluster_id) for node_id, node in sorted(nodes.items()))


def cluster_heads(manager):
    return tuple(sorted(manager.cluster_to_head.items()))


def expected_cluster_heads(nodes):
    members = defaultdict(list)
    for node_id, node in sorted(nodes.items()):
        if node.cluster_id is not None:
            members[node.cluster_id].append(node_id)
    return tuple(
        (cluster_id, max(node_ids, key=lambda node_id: (len(nodes[node_id].original_neighbors), -node_id)))
        for cluster_id, node_ids in sorted(members.items())
    )


def forwarding_graph(events):
    targets = defaultdict(list)
    for sender, target in events:
        if target not in targets[sender]:
            targets[sender].append(target)
    return tuple((sender, tuple(selected)) for sender, selected in sorted(targets.items()))


def relay_sequence(first_receptions):
    return tuple((receiver, sender) for receiver, sender, _ in first_receptions)


def run_case():
    """Build and execute one fresh DC-SoC experiment."""
    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=EPS, min_samples=MIN_SAMPLES)
    simulator = TracingSimulator(
        nodes=nodes,
        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
        seed=SEED,
        base_delay=BASE_DELAY,
        jitter=JITTER,
        cluster_manager=manager,
        experiment_name="stage3_dcsoc_s10",
        strategy_name="dcsoc",
    )
    simulator.inject_message(SOURCE_ID, TRANSACTION_ID)
    simulator.run()
    received = tuple(sorted(simulator.metrics.messages[TRANSACTION_ID].first_seen_times))
    return {
        "edges": normalized_edges(graph),
        "assignments": cluster_assignments(nodes),
        "heads": cluster_heads(manager),
        "expected_heads": expected_cluster_heads(nodes),
        "forwarding_graph": forwarding_graph(simulator.forwarding_events),
        "forwarding_events": tuple(simulator.forwarding_events),
        "received": received,
        "trace_length": len(simulator.first_receptions),
        "relay_sequence": relay_sequence(simulator.first_receptions),
    }


def print_forwarding_graph(graph):
    for sender, targets in graph:
        print(f"  {sender} -> {list(targets)}")


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S10 — Baseline deterministic and reproducible")
    print("=" * 72)

    run_a = run_case()
    run_b = run_case()

    topology_ok = run_a["edges"] == run_b["edges"]
    head_rule_ok = (
        run_a["heads"] == run_a["expected_heads"]
        and run_b["heads"] == run_b["expected_heads"]
    )
    cluster_ok = (
        run_a["assignments"] == run_b["assignments"]
        and run_a["heads"] == run_b["heads"]
        and head_rule_ok
    )
    forwarding_ok = run_a["forwarding_graph"] == run_b["forwarding_graph"]
    dissemination_ok = (
        run_a["received"] == run_b["received"]
        and run_a["trace_length"] == run_b["trace_length"]
        and run_a["relay_sequence"] == run_b["relay_sequence"]
        and run_a["forwarding_events"] == run_b["forwarding_events"]
    )

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Node count          : {N}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nTopology reproducibility:")
    print(f"  Run A edges: {list(run_a['edges'])}")
    print(f"  Run B edges: {list(run_b['edges'])}")
    print("\nResult:")
    print(f"  {pass_fail(topology_ok)}")

    print("\nCluster reproducibility:")
    print(f"  Run A assignments: {dict(run_a['assignments'])}")
    print(f"  Run B assignments: {dict(run_b['assignments'])}")
    print(f"  Run A heads      : {dict(run_a['heads'])}")
    print(f"  Run B heads      : {dict(run_b['heads'])}")
    print(f"  CH rule valid    : {pass_fail(head_rule_ok)}")
    print("\nResult:")
    print(f"  {pass_fail(cluster_ok)}")

    print("\nForwarding reproducibility:")
    print("  Run A forwarding graph:")
    print_forwarding_graph(run_a["forwarding_graph"])
    print("  Run B forwarding graph:")
    print_forwarding_graph(run_b["forwarding_graph"])
    print("\nResult:")
    print(f"  {pass_fail(forwarding_ok)}")

    print("\nDissemination reproducibility:")
    print(f"  Received nodes Run A: {list(run_a['received'])}")
    print(f"  Received nodes Run B: {list(run_b['received'])}")
    print(f"  Trace length Run A  : {run_a['trace_length']}")
    print(f"  Trace length Run B  : {run_b['trace_length']}")
    print(f"  Relay sequence equal: {pass_fail(run_a['relay_sequence'] == run_b['relay_sequence'])}")
    print(f"  Forward events equal: {pass_fail(run_a['forwarding_events'] == run_b['forwarding_events'])}")
    print("\nResult:")
    print(f"  {pass_fail(dissemination_ok)}")

    passed = topology_ok and cluster_ok and forwarding_ok and dissemination_ok
    print("\nOverall S10 result:")
    print(f"  {pass_fail(passed)}")
    assert passed, "S10 DC-SoC reproducibility validation failed."


if __name__ == "__main__":
    main()
