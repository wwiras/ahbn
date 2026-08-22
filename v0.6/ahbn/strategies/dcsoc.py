from __future__ import annotations

from typing import List

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class DCSOCStrategy(ForwardingStrategy):
    """
    DC-SoC-inspired density-clustered hybrid dissemination baseline.

    Behaviour:

        Ordinary node:
            gossip-like dissemination among physical neighbours
            belonging to the same density cluster.

        Cluster head:
            preserves the same bounded forwarding budget while
            reserving limited capacity for structured forwarding
            toward neighbouring cluster heads.

    The baseline therefore combines:

        intra-cluster gossip-like dissemination
        +
        inter-cluster structured dissemination

    This strategy deliberately contains no:

        - AHBN controller
        - EWMA
        - runtime mode switching
        - adaptive fanout
        - AHBN score/weight
        - runtime observation processing
    """

    def __init__(
        self,
        fanout: int = 3,
        inter_fanout: int = 1,
        fulfill_all_structural_children: bool = False,
    ) -> None:

        if fanout < 1:
            raise ValueError(
                "DC-SoC fanout must be >= 1"
            )

        if inter_fanout < 0:
            raise ValueError(
                "DC-SoC inter_fanout must be >= 0"
            )

        self.fanout = int(
            fanout
        )

        self.inter_fanout = int(
            inter_fanout
        )

        self.fulfill_all_structural_children = bool(
            fulfill_all_structural_children
        )

    # --------------------------------------------------------
    # Utility
    # --------------------------------------------------------

    @staticmethod
    def _sample(
        simulator,
        candidates: List[int],
        k: int,
    ) -> List[int]:
        """
        Sample up to k candidates using the simulator's seeded RNG.

        Reusing simulator.rng preserves reproducibility and ensures
        DC-SoC follows the same random-seed discipline as Gossip.
        """

        if k <= 0:
            return []

        if not candidates:
            return []

        if len(candidates) <= k:
            return candidates[:]

        return simulator.rng.sample(
            candidates,
            k,
        )

    # --------------------------------------------------------
    # Public strategy interface
    # --------------------------------------------------------

    def select_targets(
        self,
        node: Node,
        message: Message,
        simulator,
    ) -> List[int]:

        cluster_mgr = (
            simulator.cluster_manager
        )

        if cluster_mgr is None:
            return []

        if node.cluster_id is None:
            return []

        # Faithful S2 structural push: an ordinary member may only uplink
        # toward its assigned core.  It never expands the payload to an
        # independently sampled set of physical neighbours.
        if getattr(node, "dcsoc_role", "leaf") == "leaf":
            if node.node_id != message.source_id:
                return []
            parent = getattr(node, "dcsoc_parent", None)
            if parent is None or parent not in simulator.nodes:
                return []
            return [parent] if simulator.nodes[parent].is_active else []

        # Core/routing nodes drive propagation down the explicit DAG.  The
        # existing fixed fanout remains a total resource bound.
        structural_children = [
            child_id
            for child_id in getattr(node, "dcsoc_children", [])
            if child_id in simulator.nodes and simulator.nodes[child_id].is_active
            and child_id != message.source_id
        ]
        if structural_children:
            if self.fulfill_all_structural_children:
                return structural_children
            return structural_children[: self.fanout]

        # ----------------------------------------------------
        # Intra-cluster Gossip candidates.
        #
        # Only active PHYSICAL neighbours in the same density
        # cluster participate in the gossip-like component.
        # ----------------------------------------------------

        local_candidates = [
            nbr_id
            for nbr_id
            in node.neighbors

            if nbr_id
            != node.node_id

            and nbr_id
            in simulator.nodes

            and simulator.nodes[
                nbr_id
            ].is_active

            and simulator.nodes[
                nbr_id
            ].cluster_id
            == node.cluster_id
        ]

        # ----------------------------------------------------
        # Ordinary density-cluster member.
        #
        # Pure intra-cluster gossip-like dissemination.
        # ----------------------------------------------------

        if not node.is_cluster_head:

            return self._sample(
                simulator,
                local_candidates,
                min(
                    self.fanout,
                    len(
                        local_candidates
                    ),
                ),
            )

        # ----------------------------------------------------
        # Density-cluster core/head.
        #
        # In addition to local gossip-like dissemination,
        # the core/head provides the structured path between
        # density clusters.
        # ----------------------------------------------------

        gateway_candidates = [
            gateway_id
            for gateway_id
            in node.gateway_neighbors

            if gateway_id
            != node.node_id

            and gateway_id
            in simulator.nodes

            and simulator.nodes[
                gateway_id
            ].is_active
        ]

        # ----------------------------------------------------
        # Reserve a bounded portion of the SAME forwarding
        # budget for inter-cluster propagation.
        #
        # This is important for fairness: DC-SoC does not get
        # fanout + extra unlimited gateway transmissions.
        # ----------------------------------------------------

        gateway_budget = min(
            self.inter_fanout,
            self.fanout,
            len(
                gateway_candidates
            ),
        )

        selected_gateways = (
            self._sample(
                simulator,
                gateway_candidates,
                gateway_budget,
            )
        )

        # ----------------------------------------------------
        # Remaining budget performs local gossip-like
        # dissemination.
        # ----------------------------------------------------

        local_budget = max(
            0,
            self.fanout
            - len(
                selected_gateways
            ),
        )

        selected_local = (
            self._sample(
                simulator,
                local_candidates,
                min(
                    local_budget,
                    len(
                        local_candidates
                    ),
                ),
            )
        )

        # ----------------------------------------------------
        # Gateway first preserves an outward structured path.
        # Remove accidental duplicates defensively.
        # ----------------------------------------------------

        return [
            target_id

            for target_id
            in dict.fromkeys(
                selected_gateways
                + selected_local
            )

            if target_id
            != node.node_id
        ]
