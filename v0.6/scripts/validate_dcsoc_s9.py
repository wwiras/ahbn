"""Stage 3.4 S9: validate end-to-end DC-SoC dissemination."""

from collections import defaultdict, deque

import networkx as nx

from ahbn.control import NodeControlState
from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED, N, BA_M = 42, 30, 3
EPS, MIN_SAMPLES = 2.0, 3
FANOUT, INTER_FANOUT = 3, 1
BASE_DELAY, JITTER = 1.0, 0.2
SOURCE_ID, TRANSACTION_ID = 0, "1"


class TracingSimulator(Simulator):
    """Observe production sends and first receptions without changing them."""

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


def altered_control_state():
    return NodeControlState(
        d_hat=1.0,
        l_hat=0.9,
        u_hat=0.8,
        c_hat=0.7,
        score=-1000.0,
        weight=0.0,
        mode="cluster",
        fanout=4,
    )


def run_case(modify_control=False):
    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=EPS, min_samples=MIN_SAMPLES)
    if modify_control:
        for node in nodes.values():
            node.control = altered_control_state()
    simulator = TracingSimulator(
        nodes=nodes,
        strategy=DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT),
        seed=SEED,
        base_delay=BASE_DELAY,
        jitter=JITTER,
        cluster_manager=manager,
        experiment_name="stage3_dcsoc_s9",
        strategy_name="dcsoc",
    )
    simulator.inject_message(SOURCE_ID, TRANSACTION_ID)
    simulator.run()
    received = set(simulator.metrics.messages[TRANSACTION_ID].first_seen_times)
    return graph, nodes, simulator, received


def reachable_from_observed_forwards(forwarding_events):
    """Independent graph traversal over production-observed forwarding edges."""
    adjacency = defaultdict(set)
    for sender, target in forwarding_events:
        adjacency[sender].add(target)
    reachable = {SOURCE_ID}
    queue = deque([SOURCE_ID])
    while queue:
        sender = queue.popleft()
        for target in sorted(adjacency[sender]):
            if target not in reachable:
                reachable.add(target)
                queue.append(target)
    return reachable


def propagation_hops(forwarding_events):
    adjacency = defaultdict(list)
    for sender, target in forwarding_events:
        if target not in adjacency[sender]:
            adjacency[sender].append(target)
    depths = {SOURCE_ID: 0}
    queue = deque([SOURCE_ID])
    while queue:
        sender = queue.popleft()
        for target in adjacency[sender]:
            if target not in depths:
                depths[target] = depths[sender] + 1
                queue.append(target)
    grouped = defaultdict(list)
    for sender, targets in adjacency.items():
        if sender in depths:
            grouped[depths[sender]].append((sender, targets))
    return grouped


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S9 — End-to-end dissemination completes successfully")
    print("=" * 72)

    graph, nodes, run1, received1 = run_case()
    _, _, run2, received2 = run_case()
    _, _, case_b, received_b = run_case(modify_control=True)

    reachable = reachable_from_observed_forwards(run1.forwarding_events)
    missing = reachable - received1
    delivery_ok = not missing
    deterministic = received1 == received2
    independent = received1 == received_b
    origin_ok = SOURCE_ID in received1
    multi_hop = any(
        sender != SOURCE_ID for sender, _ in run1.forwarding_events
    )

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Node count          : {graph.number_of_nodes()}")
    print(f"  Edge count          : {graph.number_of_edges()}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nMessage:")
    print(f"  Transaction id      : {TRANSACTION_ID}")
    print(f"  Source node         : {SOURCE_ID}")

    print("\nPropagation trace:")
    hops = propagation_hops(run1.forwarding_events)
    for hop in sorted(hops):
        print(f"  Hop {hop}:")
        for sender, targets in sorted(hops[hop]):
            label = "source" if sender == SOURCE_ID else "relay"
            print(f"    {label}={sender} targets={targets}")

    print("\nDelivery validation:")
    print(f"  Expected reachable nodes: {sorted(reachable)}")
    print(f"  Received nodes          : {sorted(received1)}")
    print(f"  Missing nodes           : {sorted(missing)}")
    print(f"  Source originated       : {pass_fail(origin_ok)}")
    print(f"  Multi-hop observed      : {pass_fail(multi_hop)}")
    print(f"  Result                  : {pass_fail(delivery_ok)}")

    print("\nDeterminism validation:")
    print(f"  Run 1 received: {sorted(received1)}")
    print(f"  Run 2 received: {sorted(received2)}")
    print(f"  Result        : {pass_fail(deterministic)}")

    print("\nAHBN independence validation:")
    print(f"  Case A received: {sorted(received1)}")
    print(f"  Case B received: {sorted(received_b)}")
    print(f"  Result         : {pass_fail(independent)}")

    passed = origin_ok and multi_hop and delivery_ok and deterministic and independent
    print("\n" + "-" * 72)
    print(f"S9 RESULT: {pass_fail(passed)}")
    print("-" * 72)
    assert passed, "S9 end-to-end DC-SoC dissemination validation failed."


if __name__ == "__main__":
    main()
