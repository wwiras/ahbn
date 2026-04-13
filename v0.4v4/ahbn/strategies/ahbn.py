from __future__ import annotations

import hashlib
from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.cluster import ClusterStrategy
from ahbn.strategies.gossip import GossipStrategy


def deterministic_hash01(*parts: str) -> float:
    s = ":".join(parts).encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:16], 16) / float(16**16)


class AHBNStrategy(ForwardingStrategy):
    """
    Backward-compatible AHBN strategy.

    Supports:
    - Exp07 / Exp08 legacy mode:
        hard mode switch, optionally adaptive fanout
    - Exp09 upgraded hybrid mode:
        combine cluster + gossip targets using control weight and tau
    """

    def __init__(
        self,
        default_fanout: int = 3,
        adaptive_fanout: bool = False,
        hybrid_mode: bool = True,
        use_tau_gate: bool = True,
        min_cluster_targets: int = 1,
    ) -> None:
        self.default_fanout = default_fanout
        self.adaptive_fanout = adaptive_fanout
        self.hybrid_mode = hybrid_mode
        self.use_tau_gate = use_tau_gate
        self.min_cluster_targets = min_cluster_targets

        self._gossip = GossipStrategy(fanout=default_fanout)
        self._cluster = ClusterStrategy()

    def _get_effective_fanout(self, node: Node) -> int:
        if self.adaptive_fanout:
            return max(1, int(node.control.fanout))
        return max(1, int(self.default_fanout))

    def _passes_tau_gate(self, node: Node, message: Message) -> bool:
        tau = getattr(node.control, "tau", 1.0)
        score = deterministic_hash01(str(node.node_id), message.message_id)
        return score < tau

    def _dedup_preserve_order(self, targets: List[int], self_id: int) -> List[int]:
        return [t for t in dict.fromkeys(targets) if t != self_id]

    def select_targets(self, node: Node, message: Message, simulator) -> List[int]:
        fanout = self._get_effective_fanout(node)
        self._gossip.fanout = fanout

        # ------------------------------------------------------------
        # Legacy behavior for Exp07 / Exp08 compatibility if needed
        # ------------------------------------------------------------
        if not self.hybrid_mode:
            if node.control.mode == "gossip":
                return self._gossip.select_targets(node, message, simulator)
            return self._cluster.select_targets(node, message, simulator)

        # ------------------------------------------------------------
        # Exp09 hybrid behavior
        # ------------------------------------------------------------
        if self.use_tau_gate and not self._passes_tau_gate(node, message):
            return []

        gossip_weight = float(getattr(node.control, "weight", 0.5))
        gossip_weight = max(0.0, min(1.0, gossip_weight))

        # Collect candidate targets from both components
        gossip_targets = self._gossip.select_targets(node, message, simulator)
        cluster_targets = self._cluster.select_targets(node, message, simulator)

        gossip_targets = self._dedup_preserve_order(gossip_targets, node.node_id)
        cluster_targets = self._dedup_preserve_order(cluster_targets, node.node_id)

        # How many targets should come from each side?
        # weight near 1 => more gossip
        # weight near 0 => more cluster
        n_gossip = int(round(fanout * gossip_weight))
        n_cluster = fanout - n_gossip

        # Keep a minimal structural reserve in hybrid mode so AHBN
        # does not collapse into pure gossip in dense-topology Exp09.
        if fanout > 1:
            n_cluster = max(self.min_cluster_targets, n_cluster)
            n_cluster = min(n_cluster, fanout)
            n_gossip = max(0, fanout - n_cluster)

        selected: List[int] = []

        # First take structured targets
        selected.extend(cluster_targets[:n_cluster])

        # Then fill with gossip targets not already chosen
        for t in gossip_targets:
            if t not in selected:
                selected.append(t)
            if len(selected) >= fanout:
                break

        # If still short, backfill from whichever pool has leftovers
        if len(selected) < fanout:
            for t in cluster_targets:
                if t not in selected:
                    selected.append(t)
                if len(selected) >= fanout:
                    break

        if len(selected) < fanout:
            for t in gossip_targets:
                if t not in selected:
                    selected.append(t)
                if len(selected) >= fanout:
                    break

        return selected[:fanout]