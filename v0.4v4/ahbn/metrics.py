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

    # Exp10 additions
    failure_mode: Optional[str] = None
    failure_trigger_time: Optional[float] = None
    failed_node_id: Optional[int] = None

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

    # Exp10 additions
    def record_failure_trigger(
        self,
        failure_mode: str,
        trigger_time: float,
        failed_node_id: Optional[int],
    ) -> None:
        self.failure_mode = failure_mode
        self.failure_trigger_time = trigger_time
        self.failed_node_id = failed_node_id

    def summarize_message(self, message_id: str, total_nodes: int) -> dict:
        rec = self.messages[message_id]
        seen = len(rec.first_seen_times)
        delivery_ratio = seen / total_nodes if total_nodes else 0.0

        propagation_delay: Optional[float]
        end_time: Optional[float] = None
        if rec.first_seen_times:
            end_time = max(rec.first_seen_times.values())
            propagation_delay = end_time - rec.created_at
        else:
            propagation_delay = None

        recovery_time: Optional[float] = None
        if self.failure_trigger_time is not None and end_time is not None:
            recovery_time = max(0.0, end_time - self.failure_trigger_time)

        return {
            "message_id": message_id,
            "delivery_ratio": delivery_ratio,
            "propagation_delay": propagation_delay,
            "duplicates": rec.duplicate_count,
            "total_forwards": rec.total_forwards,
            "failure_mode": self.failure_mode,
            "recovery_time": recovery_time,
            "failed_node_id": self.failed_node_id,
        }