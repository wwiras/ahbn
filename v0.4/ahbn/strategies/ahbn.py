from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy


class AHBNStrategy(ForwardingStrategy):
    def __init__(self, default_fanout: int = 3, adaptive_fanout: bool = False) -> None:
        self.default_fanout = default_fanout
        self.adaptive_fanout = adaptive_fanout
        self._gossip = GossipStrategy(fanout=default_fanout)
        self._cluster = ClusterStrategy()

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        # For fanout-sweep experiments such as Exp07, keep AHBN's gossip component
        # tied to the configured experimental fanout.
        # For later adaptive experiments, allow the controller to override it.
        if self.adaptive_fanout:
            self._gossip.fanout = node.control.fanout
        else:
            self._gossip.fanout = self.default_fanout

        if node.control.mode == "gossip":
            return self._gossip.select_targets(node, message, simulator)

        return self._cluster.select_targets(node, message, simulator)