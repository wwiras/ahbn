from __future__ import annotations

from typing import List, Optional

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class GossipStrategy(ForwardingStrategy):
    """
    Pure Gossip dissemination strategy.

    The strategy performs no adaptive decision-making.

    Responsibility:
        Randomly select up to `fanout` active physical neighbors.

    AHBN may update `fanout` before calling this strategy.
    """

    def __init__(self, fanout: int = 3) -> None:
        if fanout < 1:
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
        Select up to `fanout` active neighbors uniformly at random.
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

        k = min(
            int(self.fanout),
            len(candidates),
        )

        return simulator.rng.sample(
            candidates,
            k,
        )
