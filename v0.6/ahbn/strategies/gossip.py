from __future__ import annotations

from typing import List

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

    def __init__(self, fanout: int | None = 3) -> None:
        if fanout is not None and fanout < 1:
            raise ValueError("fanout must be >= 1")

        self.fanout = fanout

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
    ) -> List[int]:
        """
        Select up to `fanout` active neighbors uniformly at random.
        """

        candidates = [
            nbr_id
            for nbr_id in node.neighbors
            if nbr_id != node.node_id
            and nbr_id in simulator.nodes
            and simulator.nodes[nbr_id].is_active
        ]

        if not candidates:
            return []

        if self.fanout is None:
            return candidates

        k = min(
            int(self.fanout),
            len(candidates),
        )

        return simulator.rng.sample(
            candidates,
            k,
        )
