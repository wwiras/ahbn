from __future__ import annotations

from ahbn.cluster import ClusterManager
from ahbn.message import Message
from ahbn.node import Node
from ahbn.simulator import Simulator
from ahbn.strategies.cluster import ClusterStrategy


def build_case(eligible_count: int) -> tuple[Simulator, Node, Message, list[int]]:
    sender_id = 0
    head_id = 1
    expected = list(range(2, 2 + eligible_count))
    physical_only = [100, 101]
    nodes = {
        node_id: Node(node_id=node_id)
        for node_id in [sender_id, head_id, *expected, *physical_only]
    }
    head = nodes[head_id]
    head.cluster_id = 0
    head.is_cluster_head = True
    head.neighbors = [sender_id, *expected, *physical_only]
    head.original_neighbors = list(head.neighbors)

    # The sender and up to five obligations are local members. Any remaining
    # obligations are structural gateway heads. Physical-only nodes must never
    # enter the Structured target set.
    local_targets = expected[:5]
    gateway_targets = expected[5:]
    head.gateway_neighbors = gateway_targets
    cluster_manager = ClusterManager(
        cluster_to_members={0: [sender_id, head_id, *local_targets]},
        cluster_to_head={0: head_id},
    )
    simulator = Simulator(
        nodes=nodes,
        strategy=ClusterStrategy(),
        seed=42,
        base_delay=1.0,
        jitter=0.0,
        cluster_manager=cluster_manager,
        controller=None,
        strategy_name="cluster",
    )
    message = Message("structured-probe", sender_id, 0.0)
    return simulator, head, message, expected


def main() -> None:
    print("S0 Structured deterministic target-count probes")
    for eligible_count in range(1, 8):
        simulator, head, message, expected = build_case(eligible_count)
        actual = simulator.strategy.select_targets(
            head,
            message,
            simulator,
            exclude_target_id=0,
        )
        assert actual == expected, (eligible_count, expected, actual)
        print(
            f"eligible={eligible_count} selected={len(actual)} "
            f"targets={actual} PASS"
        )

    simulator, _head, message, expected = build_case(7)
    simulator.handle_receive(
        now=1.0,
        dst_id=1,
        src_id=0,
        message=message,
        sent_at=0.0,
    )
    scheduled = [event.payload["dst_id"] for event in simulator.queue]
    assert sorted(scheduled) == expected, (expected, scheduled)
    assert 0 not in scheduled
    assert 100 not in scheduled and 101 not in scheduled
    print(f"event scheduling targets={sorted(scheduled)} PASS")
    print("sender exclusion=PASS physical-neighbor isolation=PASS")
    print("S0 STRUCTURED PROBE: PASS")


if __name__ == "__main__":
    main()
