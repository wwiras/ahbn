"""S1 Test F: minimal end-to-end sanity checks for four comparators."""

from __future__ import annotations

import networkx as nx

from ahbn.control import AHBNController, AHBNParams
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import (
    assign_dcsoc_clusters,
    assign_static_clusters,
    build_nodes_from_graph,
    get_dcsoc_master,
)

SEED = 42
N = 8
BA_M = 2
MESSAGE_ID = "s1-test-f"


def edge_tuple(graph):
    return tuple(sorted((min(a, b), max(a, b)) for a, b in graph.edges()))


class AuditSimulator(Simulator):
    def __init__(self, *args, **kwargs):
        self.scheduled_forwards = []
        self.repeated_receptions = []
        self.exclusion_checks = []
        super().__init__(*args, **kwargs)

    def send_message(self, src_id, dst_id, message, now):
        before = len(self.queue)
        super().send_message(src_id, dst_id, message, now)
        if len(self.queue) == before + 1:
            self.scheduled_forwards.append((src_id, dst_id, now))

    def handle_receive(self, now, dst_id, src_id, message, sent_at=None):
        duplicate_before = self.nodes[dst_id].has_seen(message.message_id)
        start = len(self.scheduled_forwards)
        super().handle_receive(now, dst_id, src_id, message, sent_at)
        if duplicate_before:
            self.repeated_receptions.append((src_id, dst_id, now))
        elif src_id != dst_id:
            selected = tuple(dst for _, dst, _ in self.scheduled_forwards[start:])
            self.exclusion_checks.append((dst_id, src_id, selected, src_id not in selected))


def fresh_graph():
    return nx.barabasi_albert_graph(N, BA_M, seed=SEED)


def build_case(name):
    graph = fresh_graph()
    nodes = build_nodes_from_graph(graph)
    manager = None
    controller = None
    source = 0

    if name == "Gossip":
        strategy = GossipStrategy(fanout=None)
    elif name == "Structured":
        manager = assign_static_clusters(nodes, num_clusters=2)
        strategy = ClusterStrategy()
        source = 2  # configured ordinary member; normal Structured injection wiring
    elif name == "DC-SoC":
        manager = assign_dcsoc_clusters(nodes, eps=2.0, min_samples=2)
        strategy = DCSOCStrategy()
        source = get_dcsoc_master(nodes)
    elif name == "AHBN":
        manager = assign_static_clusters(nodes, num_clusters=2)
        controller = AHBNController(AHBNParams())
        strategy = AHBNStrategy(default_fanout=3, adaptive_fanout=True)
    else:
        raise ValueError(name)

    simulator = AuditSimulator(
        nodes=nodes,
        strategy=strategy,
        seed=SEED,
        base_delay=1.0,
        jitter=0.0,
        cluster_manager=manager,
        controller=controller,
        experiment_name="s1_test_f",
        strategy_name=name.lower(),
    )
    return graph, simulator, source


def run_case(name):
    graph, simulator, source = build_case(name)
    repeated_graph = fresh_graph()
    edges_a = edge_tuple(graph)
    edges_b = edge_tuple(repeated_graph)
    assert edges_a == edges_b

    simulator.inject_message(source, MESSAGE_ID)
    simulator.run()

    record = simulator.metrics.messages[MESSAGE_ID]
    active = tuple(sorted(nid for nid, node in simulator.nodes.items() if node.is_active))
    delivered = tuple(sorted(record.first_seen_times))
    summary = simulator.metrics.summarize_message(MESSAGE_ID, len(active))

    origin_ok = (
        simulator.message_source_id == source
        and record.source_id == source
        and record.created_at == 0.0
        and record.first_seen_times.get(source) == 0.0
    )
    if name == "DC-SoC":
        origin_ok = origin_ok and source == get_dcsoc_master(simulator.nodes)

    expected_delivery = len(delivered) / len(active)
    expected_delay = max(record.first_seen_times.values()) - record.created_at
    delivery_ok = summary["delivery_ratio"] == expected_delivery
    duplicates_ok = summary["duplicates"] == len(simulator.repeated_receptions)
    forwards_ok = summary["total_forwards"] == len(simulator.scheduled_forwards)
    delay_ok = summary["propagation_delay"] == expected_delay
    forwarding_ok = len(simulator.scheduled_forwards) > 0
    exclusion_ok = bool(simulator.exclusion_checks) and all(
        passed for _, _, _, passed in simulator.exclusion_checks
    )

    checks = {
        "origin": origin_ok,
        "forwarding": forwarding_ok,
        "sender_exclusion": exclusion_ok,
        "delivery_accounting": delivery_ok,
        "duplicate_accounting": duplicates_ok,
        "total_forward_accounting": forwards_ok,
        "simulation_time_delay": delay_ok,
        "deterministic_topology": edges_a == edges_b,
    }
    assert all(checks.values()), (name, checks)

    print(f"\n{name}")
    print(f"  topology run A: {edges_a}")
    print(f"  topology run B: {edges_b}")
    print(f"  source/effective origin: {source}; first_seen={record.first_seen_times[source]}")
    if name == "DC-SoC":
        print(f"  active Master: {get_dcsoc_master(simulator.nodes)}")
    print(f"  forwarding events: metric={summary['total_forwards']} traced={len(simulator.scheduled_forwards)}")
    print(f"  sender-exclusion checks: {simulator.exclusion_checks}")
    print(f"  delivery: metric={summary['delivery_ratio']} unique={len(delivered)} active={len(active)}")
    print(f"  duplicates: metric={summary['duplicates']} traced={len(simulator.repeated_receptions)}")
    print(f"  propagation delay: metric={summary['propagation_delay']} event-derived={expected_delay}")
    print(f"  first-seen event times: {tuple(sorted(record.first_seen_times.items()))}")
    print(f"  checks: {checks}")


def main():
    for comparator in ("Gossip", "Structured", "DC-SoC", "AHBN"):
        run_case(comparator)
    print("\nS1 TEST F MINIMAL END-TO-END VALIDATOR: PASS")


if __name__ == "__main__":
    main()
