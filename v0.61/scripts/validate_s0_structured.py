from __future__ import annotations

from dataclasses import dataclass

from ahbn.config import load_yaml_config
from ahbn.cluster import ClusterManager
from ahbn.message import Message
from ahbn.node import Node
from ahbn.simulator import Simulator
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.topology import assign_static_clusters, build_nodes_from_graph, get_or_build_topology


@dataclass(frozen=True)
class ClusterTrace:
    condition: str
    seed: int
    heads: tuple[int, ...]
    member_counts: tuple[int, ...]
    head_degrees: tuple[int, ...]
    max_selected: int


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
    # Simulator.inject_message() normally performs this registration before
    # its initial receive event. Register directly here because this probe
    # intentionally begins at the subsequent 0 -> 1 receive.
    simulator.metrics.register_message(
        message.message_id,
        message.source_id,
        message.created_at,
    )
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

    exp08_traces = verify_exp08()
    exp09_traces = verify_exp09()

    print("============================================================")
    print("S0 — STRUCTURED FOLLOW-UP VERIFICATION")
    print("============================================================")
    print("STRUCT-19")
    print(f"  Selected targets:                       {len(expected)}")
    print(f"  Scheduled targets:                      {len(scheduled)}")
    print("  Selected == scheduled:                  PASS")
    print("  Sender excluded through scheduling:     PASS")
    print("EXP08 CH")
    print("  CH assignment identified:               PASS")
    print("  Bottleneck target == intended CH:       PASS (all CH destinations)")
    print("  Fixed-seed reproducibility:             PASS")
    print("EXP09 CH / DENSITY")
    print("  CH assignment identified:               PASS")
    print("  Active topology used:                   PASS (configured ER probabilities)")
    print("  Same seed reproduces CH:                PASS")
    print("  Strategy uses active CH state:          PASS")
    print("FANOUT")
    print("  CH structural targets truncated:        NO")
    print("Historical v0.60 exact filesystem diff:")
    print("  UNAVAILABLE (requested directory is absent)")
    print("Source repairs performed:")
    print("  YES (prior sender-exclusion dispatch repair only)")
    print("Files modified:")
    print("  ahbn/simulator.py")
    print("  scripts/validate_s0_structured.py")
    print("  docs/v0.61_S0_fix_structured.md")
    print("Documentation:")
    print("  docs/v0.61_S0_fix_structured.md")
    print(f"Verification cases: Exp08={len(exp08_traces)} Exp09={len(exp09_traces)}")
    print("FINAL:")
    print("  S0 STRUCTURED REPAIRED → PASS")
    print("============================================================")


def build_structured_trace(
    *, topology_type: str, num_nodes: int, seed: int, num_clusters: int,
    condition: str, edge_prob: float | None = None, ba_m: int | None = None,
) -> tuple[ClusterTrace, dict[int, Node], ClusterManager, ClusterStrategy]:
    graph = get_or_build_topology(
        topology_type=topology_type,
        num_nodes=num_nodes,
        seed=seed,
        use_cache=True,
        edge_prob=edge_prob,
        ba_m=ba_m,
    )
    nodes = build_nodes_from_graph(graph)
    manager = assign_static_clusters(
        nodes,
        num_clusters=num_clusters,
        resource_aware_heads=False,
    )
    strategy = ClusterStrategy()
    simulator = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        cluster_manager=manager,
        controller=None,
        strategy_name="cluster",
    )
    probe_message = Message("ch-probe", 0, 0.0)
    heads = tuple(manager.cluster_to_head[cid] for cid in sorted(manager.cluster_to_head))
    member_counts = tuple(
        len(manager.cluster_to_members[cid]) for cid in sorted(manager.cluster_to_members)
    )
    head_degrees = tuple(len(nodes[head].neighbors) for head in heads)
    selected_counts = [
        len(strategy.select_targets(nodes[head], probe_message, simulator))
        for head in heads
    ]
    assert strategy.fanout is None
    for cluster_id, head_id in manager.cluster_to_head.items():
        member = next(mid for mid in manager.cluster_to_members[cluster_id] if mid != head_id)
        assert strategy.select_targets(nodes[member], probe_message, simulator) == [head_id]
        expected_head_targets = {
            mid for mid in manager.cluster_to_members[cluster_id] if mid != head_id
        } | set(nodes[head_id].gateway_neighbors)
        actual_head_targets = set(
            strategy.select_targets(nodes[head_id], probe_message, simulator)
        )
        assert actual_head_targets == expected_head_targets
    return (
        ClusterTrace(
            condition=condition,
            seed=seed,
            heads=heads,
            member_counts=member_counts,
            head_degrees=head_degrees,
            max_selected=max(selected_counts),
        ),
        nodes,
        manager,
        strategy,
    )


def verify_exp08() -> list[ClusterTrace]:
    cfg = load_yaml_config("configs/exp08_ch_bottleneck.yaml")
    traces: list[ClusterTrace] = []
    print("Exp08 Structured CH verification")
    for factor in (cfg["ch_overload_factor"][0], cfg["ch_overload_factor"][-1]):
        trace, nodes, manager, _strategy = build_structured_trace(
            topology_type=cfg["topology_type"],
            num_nodes=cfg["num_nodes"],
            seed=cfg["seed"],
            num_clusters=cfg["num_clusters"],
            condition=f"ch_overload_factor={factor}",
            ba_m=cfg["ba_m"],
        )
        intended = set(trace.heads)
        runtime_condition_targets = {
            node_id for node_id, node in nodes.items() if node.is_cluster_head
        }
        assert runtime_condition_targets == intended == set(manager.cluster_to_head.values())
        traces.append(trace)
        print(
            f"seed={trace.seed} factor={factor} CHs={list(trace.heads)} "
            f"members={list(trace.member_counts)} degrees={list(trace.head_degrees)} "
            f"bottleneck_targets={sorted(runtime_condition_targets)} MATCH=PASS "
            f"max_CH_targets={trace.max_selected}"
        )
    assert traces[0].heads == traces[1].heads
    return traces


def verify_exp09() -> list[ClusterTrace]:
    cfg = load_yaml_config("configs/exp09_dense_topology.yaml")
    cases = [
        (cfg["edge_probs"][0], cfg["seed"]),
        (cfg["edge_probs"][-1], cfg["seed"]),
        (cfg["edge_probs"][-1], cfg["seed"]),
        (cfg["edge_probs"][-1], cfg["seed"] + 1),
    ]
    traces: list[ClusterTrace] = []
    print("Exp09 Structured CH / ER density verification")
    for edge_prob, seed in cases:
        trace, _nodes, _manager, _strategy = build_structured_trace(
            topology_type=cfg["topology_type"],
            num_nodes=cfg["num_nodes"],
            seed=seed,
            num_clusters=cfg["num_clusters"],
            condition=f"edge_prob={edge_prob}",
            edge_prob=edge_prob,
        )
        traces.append(trace)
        print(
            f"edge_prob={edge_prob} seed={seed} CHs={list(trace.heads)} "
            f"members={list(trace.member_counts)} degrees={list(trace.head_degrees)} "
            f"max_CH_targets={trace.max_selected} PASS"
        )
    assert traces[1] == traces[2]
    return traces


if __name__ == "__main__":
    main()
