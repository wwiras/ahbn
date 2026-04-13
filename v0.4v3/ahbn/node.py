from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set

from ahbn.control import NodeControlState


@dataclass
class NodeStats:
    received_new: int = 0
    received_duplicate: int = 0
    forwarded: int = 0
    dropped: int = 0
    first_receive_time: Dict[str, float] = field(default_factory=dict)
    last_receive_time: Dict[str, float] = field(default_factory=dict)


@dataclass
class Node:
    node_id: int
    neighbors: List[int] = field(default_factory=list)

    cluster_id: Optional[int] = None
    is_cluster_head: bool = False
    gateway_neighbors: List[int] = field(default_factory=list)

    seen_messages: Set[str] = field(default_factory=set)
    stats: NodeStats = field(default_factory=NodeStats)
    control: NodeControlState = field(default_factory=NodeControlState)

    def has_seen(self, message_id: str) -> bool:
        return message_id in self.seen_messages

    def mark_seen(self, message_id: str) -> None:
        self.seen_messages.add(message_id)