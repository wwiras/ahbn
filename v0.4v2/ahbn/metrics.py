from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class MessageRecord:
    message_id: str
    source_id: int
    created_at: float
    first_seen_times: Dict[int, float] = field(default_factory=dict)
    duplicate_count: int = 0
    total_forwards: int = 0


@dataclass
class MetricsCollector:
    messages: Dict[str, MessageRecord] = field(default_factory=dict)

    def register_message(self, message_id: str, source_id: int, created_at: float) -> None:
        self.messages[message_id] = MessageRecord(
            message_id=message_id,
            source_id=source_id,
            created_at=created_at,
        )

    def record_first_seen(self, node_id: int, message_id: str, time: float) -> None:
        rec = self.messages[message_id]
        rec.first_seen_times.setdefault(node_id, time)

    def record_duplicate(self, message_id: str) -> None:
        self.messages[message_id].duplicate_count += 1

    def record_forward(self, message_id: str, count: int = 1) -> None:
        self.messages[message_id].total_forwards += count

    def summarize_message(self, message_id: str, total_nodes: int) -> dict:
        rec = self.messages[message_id]
        seen = len(rec.first_seen_times)
        delivery_ratio = seen / total_nodes if total_nodes else 0.0

        propagation_delay: Optional[float]
        if rec.first_seen_times:
            end_time = max(rec.first_seen_times.values())
            propagation_delay = end_time - rec.created_at
        else:
            propagation_delay = None

        return {
            "message_id": message_id,
            "delivery_ratio": delivery_ratio,
            "propagation_delay": propagation_delay,
            "duplicates": rec.duplicate_count,
            "total_forwards": rec.total_forwards,
        }