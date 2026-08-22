from __future__ import annotations

import heapq
import math
import random
from typing import Dict, Optional

from ahbn.control import AHBNController
from ahbn.event import Event
from ahbn.message import Message
from ahbn.metrics import MetricsCollector
from ahbn.node import Node
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import (
    recluster_dcsoc,
    reinstate_dcsoc_as_leaf,
    repair_dcsoc_after_failure,
    repair_topology_after_churn,
)
from ahbn.utils import AdaptiveTraceRow


class Simulator:
    """
    Event-driven ControlSim simulator with canonical AHBN sensing.

    Canonical AHBN receives exactly four normalized local observations:

        d_t : duplicate pressure       in [0, 1]
        l_t : local latency pressure   in [0, 1]
        u_t : processing/utilization   in [0, 1]
        c_t : churn/instability        in [0, 1]

    Environment-specific measurement and normalization are implemented here.
    The AHBN controller itself remains environment-independent.
    """

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
        churn_manager=None,
        experiment_name: str = "unknown",
        strategy_name: str = "unknown",
        scenario_tag: str = "default",
        enable_adaptive_trace: bool = False,
        resource_aware_heads: bool = False,
        latency_reference: Optional[float] = None,
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
        self.failure_injector = failure_injector
        self.message_source_id: Optional[int] = None
        self.churn_manager = churn_manager
        self.resource_aware_heads = resource_aware_heads

        self.experiment_name = experiment_name
        self.strategy_name = strategy_name
        self.scenario_tag = scenario_tag
        self.enable_adaptive_trace = enable_adaptive_trace
        self.adaptive_trace_rows: list[AdaptiveTraceRow] = []

        # ------------------------------------------------------------
        # Canonical ControlSim latency reference
        # ------------------------------------------------------------
        # Use the expected normal one-hop delay as the default reference:
        #
        #     E[delay] = base_delay + jitter / 2
        #
        # The saturating normalization below maps:
        #
        #     local_delay == latency_reference  ->  l_t = 0.5
        #
        # This makes the normalization interpretable and keeps l_t bounded.
        self.latency_reference = (
            float(latency_reference)
            if latency_reference is not None
            else max(1e-9, float(base_delay) + 0.5 * float(jitter))
        )

        if self.churn_manager is not None:
            self.churn_manager.schedule_events(self)

    # ================================================================
    # Event scheduling / message transport
    # ================================================================

    def schedule_event(
        self,
        time: float,
        priority: int,
        event_type: str,
        payload: dict,
    ) -> None:
        heapq.heappush(
            self.queue,
            Event(time, priority, event_type, payload),
        )

    def send_message(
        self,
        src_id: int,
        dst_id: int,
        message: Message,
        now: float,
    ) -> None:
        src = self.nodes[src_id]
        dst = self.nodes[dst_id]

        if not src.is_active or not dst.is_active:
            return

        extra = 0.0

        # Structured bottleneck pressure.
        if dst.is_cluster_head:
            extra += self.base_delay * max(
                0.0,
                self.ch_overload_factor - 1.0,
            )

        # Explicit runtime overload pressure.
        if dst.is_overloaded:
            extra += dst.extra_delay

        # Resource heterogeneity contributes to local link/processing delay.
        extra += 0.5 * max(0.0, src.processing_delay)
        extra += 0.5 * max(0.0, dst.processing_delay)

        delay = (
            self.base_delay
            + self.rng.uniform(0.0, self.jitter)
            + extra
        )

        self.schedule_event(
            time=now + delay,
            priority=1,
            event_type="receive",
            payload={
                "dst_id": dst_id,
                "src_id": src_id,
                "message": message,
                # Important: local send timestamp allows AHBN to observe
                # one-hop delay rather than end-to-end message age.
                "sent_at": now,
            },
        )

    def inject_message(
        self,
        source_id: int,
        message_id: str,
    ) -> None:
        self.message_source_id = source_id

        msg = Message(
            message_id=message_id,
            source_id=source_id,
            created_at=self.clock,
        )

        self.metrics.register_message(
            message_id,
            source_id,
            self.clock,
        )

        self.schedule_event(
            time=self.clock,
            priority=0,
            event_type="receive",
            payload={
                "dst_id": source_id,
                "src_id": source_id,
                "message": msg,
                "sent_at": self.clock,
            },
        )

    # ================================================================
    # Canonical AHBN normalization layer
    # ================================================================

    @staticmethod
    def clamp01(value: float) -> float:
        """Clamp any scalar to the canonical observation range [0, 1]."""
        return max(0.0, min(1.0, float(value)))

    @staticmethod
    def saturating_normalize(
        value: float,
        reference: float,
    ) -> float:
        """
        Saturating normalization:

            x_hat = x / (x + reference)

        Properties:
            x = 0          -> 0
            x = reference  -> 0.5
            x -> infinity  -> 1

        This avoids hard clipping while retaining a meaningful reference point.
        """
        value = max(0.0, float(value))
        reference = max(1e-9, float(reference))

        return value / (value + reference)

    def get_duplicate_observation(
        self,
        node: Node,
    ) -> float:
        """
        d_t: local duplicate pressure.

        Raw duplicate ratio is already naturally bounded:

            duplicates / total_received

        therefore only defensive clamping is needed.
        """
        return self.clamp01(
            node.stats.duplicate_ratio_raw
        )

    def get_latency_observation(
        self,
        local_delay: float,
    ) -> float:
        """
        l_t: normalized local one-hop latency pressure.

        We deliberately use local one-hop delay, not:

            now - message.created_at

        because AHBN is intended to operate from decentralized local
        observations.

        Normalization:

            l_t = local_delay / (local_delay + latency_reference)

        Hence normal expected one-hop delay maps approximately to 0.5.
        """
        return self.clamp01(
            self.saturating_normalize(
                local_delay,
                self.latency_reference,
            )
        )

    def get_utilization_observation(
        self,
        node: Node,
    ) -> float:
        """
        u_t: normalized local forwarding / processing pressure.

        The old implementation used cumulative forwards directly:

            forwarded / capacity_score

        which grows without bound as a run progresses.

        Instead we estimate average forwarding work per received message:

            avg_forwarding = forwarded / total_received

        and normalize it by AHBN's maximum forwarding budget. Resource
        capacity then scales the pressure:

            u_t =
                (avg_forwarding / max_fanout)
                / capacity_score

        Strong nodes therefore experience less pressure for the same work,
        while weak nodes experience more pressure.
        """
        total_received = max(
            1,
            node.stats.total_received,
        )

        avg_forwarding = (
            float(node.stats.forwarded)
            / float(total_received)
        )

        if self.controller is not None:
            max_fanout = max(
                1,
                int(self.controller.params.max_fanout),
            )
        else:
            max_fanout = 1

        normalized_forwarding = (
            avg_forwarding
            / float(max_fanout)
        )

        capacity = max(
            0.25,
            float(node.capacity_score),
        )

        pressure = normalized_forwarding / capacity

        # Explicit overload should be represented as maximum processing
        # pressure even if historical forwarding volume is still low.
        if node.is_overloaded:
            pressure = max(pressure, 1.0)

        return self.clamp01(pressure)

    def get_churn_observation(
        self,
        churn_proxy: float,
    ) -> float:
        """
        c_t: local/network instability pressure.

        Exp11 already expresses churn_rate as a proportion, so it maps
        directly to [0, 1].

        Ordinary packet receptions do not manufacture a zero-valued churn
        observation. If no new churn observation exists, c_hat is retained.
        """
        return self.clamp01(churn_proxy)

    # ================================================================
    # AHBN adaptive trace
    # ================================================================

    def log_adaptive_trace(
        self,
        node: Node,
        event_type: str,
        message: Optional[Message] = None,
        duplicate_obs: Optional[float] = None,
        latency_obs: Optional[float] = None,
        utilization_obs: Optional[float] = None,
        churn_obs: Optional[float] = None,
        mode_switched: bool = False,
        fanout_changed: bool = False,
    ) -> None:

        if not self.enable_adaptive_trace or self.controller is None:
            return

        snap = self.controller.snapshot_state(
            node.control
        )

        self.adaptive_trace_rows.append(
            AdaptiveTraceRow(
                # ------------------------------------------------
                # Experiment context
                # ------------------------------------------------
                experiment=self.experiment_name,
                strategy=self.strategy_name,
                seed=self.seed,
                scenario_tag=self.scenario_tag,

                # ------------------------------------------------
                # Event context
                # ------------------------------------------------
                time=self.clock,
                node_id=node.node_id,

                message_id=(
                    message.message_id
                    if message is not None
                    else None
                ),

                event_type=event_type,

                # ------------------------------------------------
                # Raw observations
                # ------------------------------------------------
                duplicate_obs=duplicate_obs,
                latency_obs=latency_obs,
                utilization_obs=utilization_obs,
                churn_obs=churn_obs,

                # ------------------------------------------------
                # EWMA state
                # ------------------------------------------------
                d_hat=snap["d_hat"],
                l_hat=snap["l_hat"],
                u_hat=snap["u_hat"],
                c_hat=snap["c_hat"],

                # ------------------------------------------------
                # Controller computation
                # ------------------------------------------------
                score=snap["score"],
                weight=snap["weight"],

                # ------------------------------------------------
                # Controller decision
                # ------------------------------------------------
                mode=snap["mode"],
                fanout=snap["fanout"],

                # ------------------------------------------------
                # Adaptation indicators
                # ------------------------------------------------
                mode_switched=mode_switched,
                fanout_changed=fanout_changed,

                # ------------------------------------------------
                # Supporting diagnostics
                # ------------------------------------------------
                duplicate_ratio_raw=(
                    node.stats.duplicate_ratio_raw
                ),

                resource_class=(
                    node.resource_class
                ),

                capacity_score=(
                    node.capacity_score
                ),

                processing_delay=(
                    node.processing_delay
                ),

                # ------------------------------------------------
                # Cumulative counters
                # ------------------------------------------------
                received_new=(
                    node.stats.received_new
                ),

                received_duplicate=(
                    node.stats.received_duplicate
                ),

                forwarded=(
                    node.stats.forwarded
                ),
            )
        )

    # ================================================================
    # Canonical AHBN state update
    # ================================================================

    
    def update_ahbn_state(
        self,
        node: Node,
        now: float,
        local_delay: float,
        churn_proxy: Optional[float] = None,
        event_type: str = "control_update",
        message: Optional[Message] = None,
    ) -> None:
        
        if self.controller is None:
            return

        # ------------------------------------------------------------
        # Environment-specific sensing / normalization
        # ------------------------------------------------------------
        duplicate_obs = self.get_duplicate_observation(node)
        latency_obs = self.get_latency_observation(local_delay)
        utilization_obs = self.get_utilization_observation(node)
        churn_obs = (
            self.get_churn_observation(churn_proxy)
            if churn_proxy is not None
            else None
        )

        # Preserve previous outputs for adaptation-event metrics.
        prev_mode = node.control.mode
        prev_fanout = node.control.fanout

        # ------------------------------------------------------------
        # Canonical controller input
        # ------------------------------------------------------------
        self.controller.update_metrics(
            node.control,
            duplicate_obs=duplicate_obs,
            latency_obs=latency_obs,
            utilization_obs=utilization_obs,
            churn_obs=churn_obs,
        )

        # ------------------------------------------------------------
        # Canonical controller decision
        # ------------------------------------------------------------
        self.controller.decide_mode_and_fanout(
            node.control
        )

        mode_switched = (
            node.control.mode != prev_mode
        )

        fanout_changed = (
            node.control.fanout != prev_fanout
        )

        self.metrics.record_adaptation(
            mode_switched,
            fanout_changed,
        )

        
        
        # --------------------------------------------------------
        # Record exactly one trace row for this controller update
        # --------------------------------------------------------
        self.log_adaptive_trace(
            node,
            event_type=event_type,
            message=message,

            duplicate_obs=duplicate_obs,
            latency_obs=latency_obs,
            utilization_obs=utilization_obs,
            churn_obs=churn_obs,

            mode_switched=mode_switched,
            fanout_changed=fanout_changed,
        )

    # ================================================================
    # Churn feedback
    # ================================================================

    def apply_churn_feedback(
        self,
        churn_proxy: float,
    ) -> None:
        """
        Apply a churn observation to every currently active node.

        No packet was received at this instant, so duplicate, latency, and
        utilization observations should not be artificially re-sampled.
        We therefore update only the canonical churn EWMA and then recompute
        the controller decision.
        """
        if self.controller is None:
            return

        churn_obs = self.get_churn_observation(
            churn_proxy
        )

        for node in self.nodes.values():
            if not node.is_active:
                continue

            self.metrics.record_churn_feedback_update()

            prev_mode = node.control.mode
            prev_fanout = node.control.fanout

            self.controller.update_metrics(
                node.control,
                churn_obs=churn_obs,
            )

            self.controller.decide_mode_and_fanout(
                node.control
            )

            mode_switched = (
                node.control.mode != prev_mode
            )

            fanout_changed = (
                node.control.fanout != prev_fanout
            )

            self.metrics.record_adaptation(
                mode_switched,
                fanout_changed,
            )

            
            self.log_adaptive_trace(
                node,
                event_type="churn_control_update",
                # duplicate_obs=node.control.d_hat,
                # latency_obs=node.control.l_hat,
                # utilization_obs=node.control.u_hat,
                duplicate_obs=None,
                latency_obs=None,
                utilization_obs=None,
                churn_obs=churn_obs,
                mode_switched=mode_switched,
                fanout_changed=fanout_changed,
            )

    # ================================================================
    # Receive / forward processing
    # ================================================================

    def handle_receive(
        self,
        now: float,
        dst_id: int,
        src_id: int,
        message: Message,
        sent_at: Optional[float] = None,
    ) -> None:
        self.clock = now

        node = self.nodes[dst_id]

        if not node.is_active:
            return

        if sent_at is None:
            sent_at = now

        local_delay = max(
            0.0,
            now - float(sent_at),
        )

        # ------------------------------------------------------------
        # Duplicate reception
        # ------------------------------------------------------------
        if node.has_seen(message.message_id):
            node.stats.received_duplicate += 1

            self.metrics.record_duplicate(
                message.message_id
            )


            
            self.update_ahbn_state(
                node=node,
                now=now,
                local_delay=local_delay,
                event_type="duplicate_receive",
                message=message,
            )

            return

        # ------------------------------------------------------------
        # First reception
        # ------------------------------------------------------------
        node.mark_seen(message.message_id)
        node.stats.received_new += 1

        node.stats.first_receive_time.setdefault(
            message.message_id,
            now,
        )

        node.stats.last_receive_time[
            message.message_id
        ] = now

        self.metrics.record_first_seen(
            node.node_id,
            message.message_id,
            now,
        )


        
        self.update_ahbn_state(
            node=node,
            now=now,
            local_delay=local_delay,
            event_type="new_receive",
            message=message,
        )

        # ------------------------------------------------------------
        # Execute selected dissemination strategy
        # ------------------------------------------------------------
        if self.controller is not None:
            # Canonical AHBN excludes the immediate sender before applying
            # its forwarding budget.  Comparator strategies never receive
            # this AHBN-only argument because they have no controller.
            targets = self.strategy.select_targets(
                node,
                message,
                self,
                sender_id=src_id,
            )
        elif isinstance(self.strategy, GossipStrategy):
            # Standalone Gossip excludes the immediate sender before applying
            # either its Exp07 fanout or its normal all-neighbor semantics.
            targets = self.strategy.select_targets(
                node,
                message,
                self,
                exclude_target_id=src_id,
            )
        else:
            targets = self.strategy.select_targets(
                node,
                message,
                self,
            )

        unique_targets = [
            target
            for target in dict.fromkeys(targets)
            if target != node.node_id
        ]

        for target_id in unique_targets:
            self.send_message(
                node.node_id,
                target_id,
                message,
                now,
            )

        node.stats.forwarded += len(unique_targets)

        self.metrics.record_forward(
            message.message_id,
            len(unique_targets),
        )


    # ================================================================
    # Churn event handlers
    # ================================================================

    def handle_churn_leave(
        self,
        now: float,
        node_id: int,
        churn_rate: float,
    ) -> None:
        self.clock = now

        node = self.nodes[node_id]

        if not node.is_active:
            return

        was_core = node.dcsoc_role == "core"
        node.leave_network()
        if self.strategy.__class__.__name__ == "DCSOCStrategy":
            self.handle_dcsoc_failure(node_id, was_core=was_core)
        else:
            repair_topology_after_churn(
                self.nodes, self.cluster_manager,
                resource_aware_heads=self.resource_aware_heads,
            )

        self.metrics.record_churn_event("leave")
        self.metrics.record_cluster_repair()

        self.apply_churn_feedback(
            churn_proxy=churn_rate
        )

    def handle_churn_join(
        self,
        now: float,
        node_id: int,
        churn_rate: float,
    ) -> None:
        self.clock = now

        node = self.nodes[node_id]

        if node.is_active:
            return

        node.rejoin_network()
        if self.strategy.__class__.__name__ == "DCSOCStrategy":
            from ahbn.topology import refresh_active_neighbors
            refresh_active_neighbors(self.nodes)
            reinstate_dcsoc_as_leaf(self.nodes, self.cluster_manager, node_id)
            missing = sorted(mid for mid in self.metrics.messages if not node.has_seen(mid))
            if missing:
                source_id = next((
                    nid for nid in sorted(self.nodes)
                    if self.nodes[nid].is_active
                    and all(self.nodes[nid].has_seen(mid) for mid in missing)
                ), None)
                if source_id is not None:
                    self.cluster_manager.recovery_count += 1
                    self.cluster_manager.recovery_request_count += 1
                    delay = self.base_delay + self.rng.uniform(0.0, self.jitter)
                    self.schedule_event(
                        time=now + delay, priority=0, event_type="dcsoc_recovery",
                        payload={"node_id": node_id, "message_ids": missing, "started_at": now},
                    )
        else:
            repair_topology_after_churn(
                self.nodes, self.cluster_manager,
                resource_aware_heads=self.resource_aware_heads,
            )

        self.metrics.record_churn_event("join")
        self.metrics.record_cluster_repair()

        self.apply_churn_feedback(
            churn_proxy=churn_rate
        )

    # ================================================================
    # Resource metrics (evaluation only, not controller inputs)
    # ================================================================

    def get_resource_metrics(self) -> dict:
        active_nodes = [
            node
            for node in self.nodes.values()
            if node.is_active
        ]

        if not active_nodes:
            return {
                "max_normalized_load": 0.0,
                "load_balance_cv": 0.0,
                "strong_forward_share": 0.0,
                "medium_forward_share": 0.0,
                "weak_forward_share": 0.0,
            }

        norm_loads = [
            node.stats.forwarded
            / max(0.25, node.capacity_score)
            for node in active_nodes
        ]

        mean_load = (
            sum(norm_loads)
            / len(norm_loads)
        )

        variance = (
            sum(
                (value - mean_load) ** 2
                for value in norm_loads
            )
            / len(norm_loads)
        )

        std_load = math.sqrt(variance)

        total_forwarded = sum(
            node.stats.forwarded
            for node in active_nodes
        )

        class_totals = {
            "strong": 0,
            "medium": 0,
            "weak": 0,
        }

        for node in active_nodes:
            class_totals[node.resource_class] = (
                class_totals.get(
                    node.resource_class,
                    0,
                )
                + node.stats.forwarded
            )

        denom = (
            total_forwarded
            if total_forwarded > 0
            else 1
        )

        return {
            "max_normalized_load": (
                max(norm_loads)
                if norm_loads
                else 0.0
            ),
            "load_balance_cv": (
                std_load / mean_load
                if mean_load > 0
                else 0.0
            ),
            "strong_forward_share": (
                class_totals.get("strong", 0)
                / denom
            ),
            "medium_forward_share": (
                class_totals.get("medium", 0)
                / denom
            ),
            "weak_forward_share": (
                class_totals.get("weak", 0)
                / denom
            ),
        }

    # ================================================================
    # Main event loop
    # ================================================================

    def run(self, until: float = 1000.0) -> None:
        while self.queue:
            event = heapq.heappop(self.queue)

            if event.time > until:
                break

            self.clock = event.time

            if (
                self.failure_injector is not None
                and self.failure_injector.should_trigger(
                    self.clock
                )
            ):
                self.failure_injector.apply(self)

            if (
                self.failure_injector is not None
                and self.failure_injector.should_clear_overload(
                    self.clock
                )
            ):
                self.failure_injector.clear(self)

            if event.event_type == "receive":
                self.handle_receive(
                    now=event.time,
                    dst_id=event.payload["dst_id"],
                    src_id=event.payload["src_id"],
                    message=event.payload["message"],
                    sent_at=event.payload.get(
                        "sent_at",
                        event.time,
                    ),
                )

            elif event.event_type == "churn_leave":
                self.handle_churn_leave(
                    now=event.time,
                    node_id=event.payload["node_id"],
                    churn_rate=float(
                        event.payload.get(
                            "churn_rate",
                            0.0,
                        )
                    ),
                )

            elif event.event_type == "churn_join":
                self.handle_churn_join(
                    now=event.time,
                    node_id=event.payload["node_id"],
                    churn_rate=float(
                        event.payload.get(
                            "churn_rate",
                            0.0,
                        )
                    ),
                )

            elif event.event_type == "dcsoc_recovery":
                node = self.nodes[event.payload["node_id"]]
                if node.is_active:
                    for message_id in event.payload["message_ids"]:
                        node.mark_seen(message_id)
                    self.cluster_manager.recovery_transfer_count += 1
                    self.cluster_manager.recovery_time += event.time - event.payload["started_at"]

            elif event.event_type == "dcsoc_recluster":
                recluster_dcsoc(
                    self.nodes, self.cluster_manager,
                    eps=float(event.payload.get("eps", 2.0)),
                    min_samples=int(event.payload.get("min_samples", 3)),
                )
    def handle_dcsoc_failure(self, node_id: int, was_core: bool) -> Optional[int]:
        if self.cluster_manager is None or not hasattr(self.cluster_manager, "structural_edges"):
            return None
        return repair_dcsoc_after_failure(
            self.nodes, self.cluster_manager, node_id, was_core=was_core
        )
