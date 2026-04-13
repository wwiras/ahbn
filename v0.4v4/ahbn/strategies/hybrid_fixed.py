from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class HybridFixedStrategy(ForwardingStrategy):
    def __init__(self, fanout: int = 3) -> None:
        self.fanout = fanout

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        cluster_mgr = simulator.cluster_manager
        if cluster_mgr is None:
            return []

        candidates: List[int] = []

        same_cluster = [
            n for n in node.neighbors
            if simulator.nodes[n].cluster_id == node.cluster_id
        ]

        other_cluster = [
            n for n in node.neighbors
            if simulator.nodes[n].cluster_id != node.cluster_id
        ]

        if node.is_cluster_head:
            members = cluster_mgr.get_cluster_members(
                node.cluster_id, exclude=node.node_id
            )
            gateways = list(node.gateway_neighbors)

            # Structured spreading
            candidates.extend(members)
            candidates.extend(gateways)

        else:
            ch_id = cluster_mgr.get_cluster_head(node.cluster_id)

            # Priority 1: send to cluster head
            if ch_id is not None and ch_id != node.node_id:
                candidates.append(ch_id)

            # Priority 2: local cluster spread
            candidates.extend(same_cluster)

            # Priority 3: limited external spread
            candidates.extend(other_cluster)

        # Remove duplicates
        unique_candidates = [
            t for t in dict.fromkeys(candidates)
            if t != node.node_id
        ]

        if not unique_candidates:
            return []

        k = min(self.fanout, len(unique_candidates))
        return unique_candidates[:k]