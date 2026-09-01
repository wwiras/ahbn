from __future__ import annotations

from typing import List, Optional

from ahbn.message import Message
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class DCSOCStrategy(ForwardingStrategy):
    """
    DC-SoC-inspired density-clustered hybrid dissemination baseline.

    Dissemination follows the explicit DC-SoC structural DAG built by
    ``assign_dcsoc_clusters``.  Structural relationships are forwarding
    obligations and are therefore not subject to a numeric fanout budget.

    This strategy deliberately contains no:

        - AHBN controller
        - EWMA
        - runtime mode switching
        - adaptive fanout
        - AHBN score/weight
        - runtime observation processing
    """

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
            parent = getattr(node, "dcsoc_parent", None)
            if (
                parent is None
                or parent == exclude_target_id
                or parent not in simulator.nodes
            ):
                return []
            return [parent] if simulator.nodes[parent].is_active else []

        # Core/routing nodes drive propagation down the explicit DAG.  Every
        # active child is a structural forwarding obligation; only the
        # immediate sender is excluded to avoid a redundant return.
        return [
            child_id
            for child_id in getattr(node, "dcsoc_children", [])
            if (
                child_id != exclude_target_id
                and child_id in simulator.nodes
                and simulator.nodes[child_id].is_active
            )
        ]
