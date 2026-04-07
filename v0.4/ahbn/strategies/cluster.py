from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class ClusterStrategy(ForwardingStrategy):
    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        cluster_mgr = simulator.cluster_manager
        if cluster_mgr is None:
            return []

        if node.is_cluster_head:
            members = cluster_mgr.get_cluster_members(node.cluster_id, exclude=node.node_id)
            gateways = list(node.gateway_neighbors)
            return list(dict.fromkeys(members + gateways))

        ch_id = cluster_mgr.get_cluster_head(node.cluster_id)
        if ch_id is not None and ch_id != node.node_id:
            return [ch_id]
        return []