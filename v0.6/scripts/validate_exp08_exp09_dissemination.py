"""Validate the narrowly scoped Exp08/Exp09 dissemination correction."""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.strategies.gossip import GossipStrategy


def active_node(node_id: int) -> Node:
    node = Node(node_id=node_id)
    node.is_active = True
    return node


def main() -> int:
    nodes = {node_id: active_node(node_id) for node_id in range(7)}
    source = nodes[0]
    source.neighbors = [1, 2, 3, 4]
    nodes[4].is_active = False
    simulator = SimpleNamespace(nodes=nodes, rng=__import__("random").Random(42))
    message = Message(message_id="m1", source_id=0, created_at=0.0)

    uncapped_gossip = GossipStrategy(fanout=None).select_targets(source, message, simulator)
    capped_gossip = GossipStrategy(fanout=2).select_targets(source, message, simulator)

    source.cluster_id = 0
    source.dcsoc_role = "core"
    source.dcsoc_children = [1, 2, 3, 4]
    simulator.cluster_manager = object()
    uncapped_dcsoc = DCSOCStrategy(
        fanout=2, fulfill_all_structural_children=True
    ).select_targets(source, message, simulator)
    capped_dcsoc = DCSOCStrategy(fanout=2).select_targets(source, message, simulator)

    checks = {
        "gossip_all_active_physical_neighbors": uncapped_gossip == [1, 2, 3],
        "gossip_default_cap_preserved": len(capped_gossip) == 2,
        "dcsoc_all_active_structural_children": uncapped_dcsoc == [1, 2, 3],
        "dcsoc_default_cap_preserved": capped_dcsoc == [1, 2],
    }
    for name, passed in checks.items():
        print(f"{name}: {'PASS' if passed else 'FAIL'}")
    passed = all(checks.values())
    print(f"EXP08/EXP09 DISSEMINATION VALIDATION: {'PASS' if passed else 'FAIL'}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
