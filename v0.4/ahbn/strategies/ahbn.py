from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy


class AHBNStrategy(ForwardingStrategy):
    def __init__(self, default_fanout: int = 3) -> None:
        self._gossip = GossipStrategy(fanout=default_fanout)
        self._cluster = ClusterStrategy()

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        self._gossip.fanout = node.control.fanout

        if node.control.mode == "gossip":
            return self._gossip.select_targets(node, message, simulator)

        return self._cluster.select_targets(node, message, simulator)