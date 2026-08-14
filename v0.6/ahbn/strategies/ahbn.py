from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy


class AHBNStrategy(ForwardingStrategy):
    """
    Canonical AHBN dissemination strategy.

    Responsibility:
        Execute the dissemination mode and fanout selected by
        the canonical AHBN controller.

    The controller decides:
        - mode: "gossip" or "cluster"
        - fanout: bounded forwarding budget

    This strategy does NOT:
        - mix Gossip and Structured targets
        - use controller weight to construct target mixtures
        - apply tau-based suppression
        - perform experiment-specific resource-aware targeting
        - override the controller decision
    """

    def __init__(
        self,
        default_fanout: int = 3,
        adaptive_fanout: bool = True,
    ) -> None:
        self.default_fanout = default_fanout
        self.adaptive_fanout = adaptive_fanout

        self._gossip = GossipStrategy(
            fanout=default_fanout
        )

        self._cluster = ClusterStrategy()

    # --------------------------------------------------------
    # Fanout
    # --------------------------------------------------------

    def _get_effective_fanout(
        self,
        node: Node,
    ) -> int:
        """
        Obtain the forwarding budget selected by AHBN.

        If adaptive fanout is enabled:
            use node.control.fanout.

        Otherwise:
            use the configured default fanout.
        """

        if self.adaptive_fanout:
            return max(
                1,
                int(node.control.fanout),
            )

        return max(
            1,
            int(self.default_fanout),
        )

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    @staticmethod
    def _dedup_preserve_order(
        targets: List[int],
        self_id: int,
    ) -> List[int]:
        """
        Remove duplicates while preserving target order,
        and prevent forwarding to the current node itself.
        """

        return [
            target
            for target in dict.fromkeys(targets)
            if target != self_id
        ]

    # --------------------------------------------------------
    # Canonical forwarding decision
    # --------------------------------------------------------

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
    ) -> List[int]:

        fanout = self._get_effective_fanout(node)

        mode = getattr(
            node.control,
            "mode",
            "gossip",
        )

        if mode == "gossip":

            self._gossip.fanout = fanout

            targets = self._gossip.select_targets(
                node,
                message,
                simulator,
            )

        elif mode == "cluster":

            self._cluster.fanout = fanout

            targets = self._cluster.select_targets(
                node,
                message,
                simulator,
            )

        else:
            raise ValueError(
                f"Unknown AHBN dissemination mode: {mode}"
            )

        return self._dedup_preserve_order(
            targets,
            node.node_id,
        )