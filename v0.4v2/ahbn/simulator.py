from __future__ import annotations

import heapq
import random
from typing import Dict, Optional

from ahbn.control import AHBNController
from ahbn.event import Event
from ahbn.message import Message
from ahbn.metrics import MetricsCollector
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy


class Simulator:
    def __init__(
        self,
        nodes: Dict[int, Node],
        strategy: ForwardingStrategy,
        seed: int = 42,
        base_delay: float = 1.0,
        jitter: float = 0.2,
        cluster_manager=None,
        controller: Optional[AHBNController] = None,
        ch_overload_factor: float = 1.0,
    ) -> None:
        self.nodes = nodes
        self.strategy = strategy
        self.seed = seed
        self.rng = random.Random(seed)
        self.clock = 0.0
        self.queue: list[Event] = []
        self.metrics = MetricsCollector()

        self.base_delay = base_delay
        self.jitter = jitter
        self.cluster_manager = cluster_manager
        self.controller = controller
        self.ch_overload_factor = ch_overload_factor

    def schedule_event(self, time: float, priority: int, event_type: str, payload: dict) -> None:
        heapq.heappush(self.queue, Event(time, priority, event_type, payload))

    def send_message(self, src_id: int, dst_id: int, message: Message, now: float) -> None:
        dst = self.nodes[dst_id]
        extra = 0.0

        if dst.is_cluster_head:
            extra = self.base_delay * max(0.0, self.ch_overload_factor - 1.0)

        delay = self.base_delay + self.rng.uniform(0.0, self.jitter) + extra
        self.schedule_event(
            time=now + delay,
            priority=1,
            event_type="receive",
            payload={"dst_id": dst_id, "src_id": src_id, "message": message},
        )

    def inject_message(self, source_id: int, message_id: str) -> None:
        msg = Message(message_id=message_id, source_id=source_id, created_at=self.clock)
        self.metrics.register_message(message_id, source_id, self.clock)
        self.schedule_event(
            time=self.clock,
            priority=0,
            event_type="receive",
            payload={"dst_id": source_id, "src_id": source_id, "message": msg},
        )

    def update_ahbn_state(self, node: Node, now: float, receive_lag: float) -> None:
        if self.controller is None:
            return

        total_recv = node.stats.received_new + node.stats.received_duplicate
        duplicate_ratio = (
            node.stats.received_duplicate / total_recv if total_recv > 0 else 0.0
        )
        load_proxy = float(node.stats.forwarded)
        latency_proxy = receive_lag

        self.controller.update_metrics(
            node.control,
            duplicate_ratio=duplicate_ratio,
            load_proxy=load_proxy,
            latency_proxy=latency_proxy,
            churn_proxy=0.0,
        )
        self.controller.decide_mode_and_fanout(node.control)

    def handle_receive(self, now: float, dst_id: int, src_id: int, message: Message) -> None:
        self.clock = now
        node = self.nodes[dst_id]

        if node.has_seen(message.message_id):
            node.stats.received_duplicate += 1
            self.metrics.record_duplicate(message.message_id)
            self.update_ahbn_state(node, now, now - message.created_at)
            return

        node.mark_seen(message.message_id)
        node.stats.received_new += 1
        node.stats.first_receive_time.setdefault(message.message_id, now)
        node.stats.last_receive_time[message.message_id] = now
        self.metrics.record_first_seen(node.node_id, message.message_id, now)

        self.update_ahbn_state(node, now, now - message.created_at)

        targets = self.strategy.select_targets(node, message, self)
        unique_targets = [t for t in dict.fromkeys(targets) if t != node.node_id]

        for t in unique_targets:
            self.send_message(node.node_id, t, message, now)

        node.stats.forwarded += len(unique_targets)
        self.metrics.record_forward(message.message_id, len(unique_targets))

    def run(self, until: float = 1000.0) -> None:
        while self.queue:
            event = heapq.heappop(self.queue)
            if event.time > until:
                break

            if event.event_type == "receive":
                self.handle_receive(
                    now=event.time,
                    dst_id=event.payload["dst_id"],
                    src_id=event.payload["src_id"],
                    message=event.payload["message"],
                )