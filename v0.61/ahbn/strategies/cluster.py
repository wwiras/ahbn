from __future__ import annotations

from typing import List, Optional

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class ClusterStrategy(ForwardingStrategy):
    """
    Pure structured / cluster dissemination strategy.

    Structural semantics:

    Member node:
        forward toward its cluster head.

    Cluster head:
        disseminate to:
            1. neighboring cluster heads / gateways
            2. members of its own cluster

    No experiment-specific adaptation is implemented here.

    The optional forwarding budget is used when this strategy
    is executed by canonical AHBN. If no budget is supplied,
    all valid structured targets are returned, preserving the
    standalone Structured baseline behaviour.
    """

    def __init__(
        self,
        fanout: Optional[int] = None,
    ) -> None:
        self.fanout = fanout

    # --------------------------------------------------------
    # Utility functions
    # --------------------------------------------------------

    @staticmethod
    def _dedup_preserve_order(
        targets: List[int],
        self_id: int,
    ) -> List[int]:
        return [
            target
            for target in dict.fromkeys(targets)
            if target != self_id
        ]

    @staticmethod
    def _is_active_target(
        target_id: int,
        simulator,
    ) -> bool:
        target = simulator.nodes.get(target_id)

        return (
            target is not None
            and target.is_active
        )

    # --------------------------------------------------------
    # Cluster-head forwarding
    # --------------------------------------------------------

    def _select_head_targets(
        self,
        node: Node,
        simulator,
        exclude_target_id: Optional[int] = None,
    ) -> List[int]:
        """
        Select structured targets for a cluster head.

        When no forwarding budget is specified:
            return all members + gateway heads.

        When a forwarding budget is specified:
            preserve both structural responsibilities where
            possible:

                - at least one inter-cluster gateway path
                - remaining budget for local members

        This is structural allocation, not adaptive control.
        """

        cluster_mgr = simulator.cluster_manager

        members = [
            member_id
            for member_id in cluster_mgr.get_cluster_members(
                node.cluster_id,
                exclude=node.node_id,
            )
            if self._is_active_target(
                member_id,
                simulator,
            )
            and member_id != exclude_target_id
        ]

        gateways = [
            gateway_id
            for gateway_id in node.gateway_neighbors
            if gateway_id != node.node_id
            and gateway_id != exclude_target_id
            and self._is_active_target(
                gateway_id,
                simulator,
            )
        ]

        members = self._dedup_preserve_order(
            members,
            node.node_id,
        )

        gateways = self._dedup_preserve_order(
            gateways,
            node.node_id,
        )

        # ----------------------------------------------------
        # Standalone Structured baseline
        # ----------------------------------------------------

        if self.fanout is None:
            return self._dedup_preserve_order(
                members + gateways,
                node.node_id,
            )

        # ----------------------------------------------------
        # AHBN bounded Structured execution
        # ----------------------------------------------------

        budget = max(
            1,
            int(self.fanout),
        )

        selected: List[int] = []

        # Preserve one outward inter-cluster path whenever
        # one exists.
        if gateways and budget > 0:
            selected.append(gateways[0])

        # Use remaining budget for local cluster dissemination.
        for member_id in members:
            if len(selected) >= budget:
                break

            if member_id not in selected:
                selected.append(member_id)

        # If capacity remains, include additional gateway paths.
        for gateway_id in gateways[1:]:
            if len(selected) >= budget:
                break

            if gateway_id not in selected:
                selected.append(gateway_id)

        return selected[:budget]

    # --------------------------------------------------------
    # Member forwarding
    # --------------------------------------------------------

    def _select_member_targets(
        self,
        node: Node,
        simulator,
        exclude_target_id: Optional[int] = None,
    ) -> List[int]:
        """
        A normal cluster member forwards only toward its
        cluster head.

        No backup peer is added.
        """

        cluster_mgr = simulator.cluster_manager

        ch_id = cluster_mgr.get_cluster_head(
            node.cluster_id
        )

        if ch_id is None:
            return []

        if ch_id == node.node_id:
            return []

        if ch_id == exclude_target_id:
            return []

        if not self._is_active_target(
            ch_id,
            simulator,
        ):
            return []

        return [ch_id]

    # --------------------------------------------------------
    # Public strategy interface
    # --------------------------------------------------------

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
        exclude_target_id: Optional[int] = None,
    ) -> List[int]:

        cluster_mgr = simulator.cluster_manager

        if cluster_mgr is None:
            return []

        if node.is_cluster_head:
            return self._select_head_targets(
                node,
                simulator,
                exclude_target_id=exclude_target_id,
            )

        return self._select_member_targets(
            node,
            simulator,
            exclude_target_id=exclude_target_id,
        )
