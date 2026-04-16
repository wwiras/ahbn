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
        failure_injector=None,
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

        # Exp10 additions
        self.failure_injector = failure_injector
        self.message_source_id: Optional[int] = None

    def schedule_event(self, time: float, priority: int, event_type: str, payload: dict) -> None:
        heapq.heappush(self.queue, Event(time, priority, event_type, payload))

    def send_message(self, src_id: int, dst_id: int, message: Message, now: float) -> None:
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]

        if not src.is_active or not dst.is_active:
            return

        extra = 0.0

        if dst.is_cluster_head:
            extra += self.base_delay * max(0.0, self.ch_overload_factor - 1.0)

        if dst.is_overloaded:
            extra += dst.extra_delay

        delay = self.base_delay + self.rng.uniform(0.0, self.jitter) + extra
        self.schedule_event(
            time=now + delay,
            priority=1,
            event_type="receive",
            payload={"dst_id": dst_id, "src_id": src_id, "message": message},
        )

    def inject_message(self, source_id: int, message_id: str) -> None:
        self.message_source_id = source_id

        msg = Message(message_id=message_id, source_id=source_id, created_at=self.clock)
        self.metrics.register_message(message_id, source_id, self.clock)
        self.schedule_event(
            time=self.clock,
            priority=0,
            event_type="receive",
            payload={"dst_id": source_id, "src_id": source_id, "message": msg},
        )

    # ------------------------------------------------------------------
    # Exp09 helpers: local topology signals
    # ------------------------------------------------------------------
    def get_node_degree(self, node: Node) -> int:
        return len(node.neighbors)

    def get_neighbor_overlap(self, node: Node) -> float:
        nbrs = set(node.neighbors)
        if not nbrs:
            return 0.0

        vals: list[float] = []
        for nbr_id in nbrs:
            nbr_node = self.nodes.get(nbr_id)
            if nbr_node is None:
                continue
            nbr_set = set(nbr_node.neighbors)
            inter = len(nbrs & nbr_set)
            union = len(nbrs | nbr_set)
            vals.append(inter / union if union > 0 else 0.0)

        if not vals:
            return 0.0
        return sum(vals) / len(vals)

    def get_redundancy_proxy(self, node: Node) -> float:
        if self.controller is None:
            return 0.0

        degree = self.get_node_degree(node)
        overlap = self.get_neighbor_overlap(node)

        degree_ref = getattr(self.controller, "degree_ref", 10.0)
        b_degree = getattr(self.controller, "b_degree", 0.25)
        b_overlap = getattr(self.controller, "b_overlap", 0.75)

        norm_degree = min(2.0, degree / max(1.0, degree_ref))
        return b_overlap * overlap + b_degree * norm_degree

    def update_ahbn_state(self, node: Node, now: float, receive_lag: float) -> None:
        if self.controller is None:
            return

        total_recv = node.stats.received_new + node.stats.received_duplicate
        duplicate_ratio = (
            node.stats.received_duplicate / total_recv if total_recv > 0 else 0.0
        )

        load_proxy = float(node.stats.forwarded)
        latency_proxy = receive_lag

        degree_proxy = float(self.get_node_degree(node))
        overlap_proxy = self.get_neighbor_overlap(node)
        redundancy_proxy = self.get_redundancy_proxy(node)

        self.controller.update_metrics(
            node.control,
            duplicate_ratio=duplicate_ratio,
            load_proxy=load_proxy,
            latency_proxy=latency_proxy,
            churn_proxy=0.0,
            degree_proxy=degree_proxy,
            overlap_proxy=overlap_proxy,
            redundancy_proxy=redundancy_proxy,
        )
        self.controller.decide_mode_and_fanout(node.control)

    def handle_receive(self, now: float, dst_id: int, src_id: int, message: Message) -> None:
        self.clock = now
        node = self.nodes[dst_id]

        if not node.is_active:
            return

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

            self.clock = event.time

            if self.failure_injector is not None and self.failure_injector.should_trigger(self.clock):
                self.failure_injector.apply(self)

            if self.failure_injector is not None and self.failure_injector.should_clear_overload(self.clock):
                self.failure_injector.clear(self)

            if event.event_type == "receive":
                self.handle_receive(
                    now=event.time,
                    dst_id=event.payload["dst_id"],
                    src_id=event.payload["src_id"],
                    message=event.payload["message"],
                )