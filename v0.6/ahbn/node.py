from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ahbn.control import NodeControlState


# ============================================================
# Per-node dissemination statistics
# ============================================================

@dataclass
class NodeStats:
    received_new: int = 0
    received_duplicate: int = 0
    forwarded: int = 0
    dropped: int = 0

    first_receive_time: Dict[str, float] = field(
        default_factory=dict
    )

    last_receive_time: Dict[str, float] = field(
        default_factory=dict
    )

    @property
    def total_received(self) -> int:
        return (
            self.received_new
            + self.received_duplicate
        )

    @property
    def duplicate_ratio_raw(self) -> float:
        """
        Local duplicate ratio:

            duplicate messages
            ------------------
            all received messages

        Naturally bounded in [0, 1].

        This raw local statistic is later used by simulator.py
        to construct the canonical duplicate observation.
        """

        total = self.total_received

        if total <= 0:
            return 0.0

        return (
            self.received_duplicate
            / total
        )


# ============================================================
# Network node
# ============================================================

@dataclass
class Node:
    node_id: int

    # --------------------------------------------------------
    # Physical overlay
    # --------------------------------------------------------

    neighbors: List[int] = field(
        default_factory=list
    )

    original_neighbors: List[int] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Structured overlay
    # --------------------------------------------------------

    cluster_id: Optional[int] = None
    is_cluster_head: bool = False

    gateway_neighbors: List[int] = field(
        default_factory=list
    )

    # --------------------------------------------------------
    # Message state
    # --------------------------------------------------------

    seen_messages: Set[str] = field(
        default_factory=set
    )

    stats: NodeStats = field(
        default_factory=NodeStats
    )

    # --------------------------------------------------------
    # Canonical AHBN controller state
    # --------------------------------------------------------

    control: NodeControlState = field(
        default_factory=NodeControlState
    )

    # --------------------------------------------------------
    # Runtime availability / overload state
    # --------------------------------------------------------

    is_active: bool = True
    is_overloaded: bool = False
    extra_delay: float = 0.0

    # --------------------------------------------------------
    # Environment resource characteristics
    # --------------------------------------------------------
    #
    # These values describe the simulated node/environment.
    # They are NOT separate AHBN controller inputs.
    #
    # simulator.py may use them when deriving normalized
    # utilization and latency pressure.
    # --------------------------------------------------------

    resource_class: str = "medium"
    capacity_score: float = 1.0
    processing_delay: float = 0.0

    # --------------------------------------------------------
    # Initialization
    # --------------------------------------------------------

    def __post_init__(self) -> None:
        if not self.original_neighbors:
            self.original_neighbors = list(
                self.neighbors
            )

    # --------------------------------------------------------
    # Message bookkeeping
    # --------------------------------------------------------

    def has_seen(
        self,
        message_id: str,
    ) -> bool:

        return (
            message_id
            in self.seen_messages
        )

    def mark_seen(
        self,
        message_id: str,
    ) -> None:

        self.seen_messages.add(
            message_id
        )

    # --------------------------------------------------------
    # Failure / recovery
    # --------------------------------------------------------

    def fail(self) -> None:
        """
        Mark the node unavailable.

        Physical and structured forwarding links are removed
        from the active node state.
        """

        self.is_active = False
        self.neighbors = []
        self.gateway_neighbors = []
        self.is_cluster_head = False

    def recover(self) -> None:
        """
        Reactivate the node.

        Topology restoration, where required, is handled by
        the simulator / experiment logic.
        """

        self.is_active = True

    # --------------------------------------------------------
    # Processing overload
    # --------------------------------------------------------

    def set_overload(
        self,
        extra_delay: float,
    ) -> None:

        self.is_overloaded = True

        self.extra_delay = max(
            0.0,
            extra_delay,
        )

    def clear_overload(self) -> None:
        self.is_overloaded = False
        self.extra_delay = 0.0

    # --------------------------------------------------------
    # Churn helpers
    # --------------------------------------------------------

    def leave_network(self) -> None:
        self.fail()

    def rejoin_network(self) -> None:
        self.recover()