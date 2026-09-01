from __future__ import annotations

from typing import List, Optional

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class GossipStrategy(ForwardingStrategy):
    """
    Pure Gossip dissemination strategy.

    The strategy performs no adaptive decision-making.

    A configured ``fanout`` bounds random forwarding (the Exp07 sweep).
    With ``fanout=None``, Gossip forwards to every eligible active physical
    neighbor (the normal Exp08/Exp09 comparator semantics).
    """

    def __init__(self, fanout: Optional[int] = None) -> None:
        if fanout is not None and fanout < 1:
            raise ValueError("fanout must be >= 1")

        self.fanout = fanout

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
        exclude_target_id: Optional[int] = None,
    ) -> List[int]:
        """
        Select all eligible neighbors, or sample up to a configured fanout.
        """

        candidates = [
            nbr_id
            for nbr_id in node.neighbors
            if nbr_id != node.node_id
            and nbr_id != exclude_target_id
            and nbr_id in simulator.nodes
            and simulator.nodes[nbr_id].is_active
        ]

        if not candidates:
            return []

        if self.fanout is None:
            return candidates

        k = min(int(self.fanout), len(candidates))

        return simulator.rng.sample(
            candidates,
            k,
        )
