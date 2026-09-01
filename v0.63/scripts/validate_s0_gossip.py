from __future__ import annotations

from pathlib import Path
import sys
from unittest.mock import patch

import networkx as nx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ahbn.config import load_yaml_config  # noqa: E402
from ahbn.message import Message  # noqa: E402
from ahbn.simulator import Simulator  # noqa: E402
from ahbn.strategies.gossip import GossipStrategy  # noqa: E402
from ahbn.topology import build_nodes_from_graph  # noqa: E402
from run_batch import exp07, exp08, exp09, run_single  # noqa: E402


def direct_probe(fanout: int | None) -> list[int]:
    graph = nx.star_graph(6)
    nodes = build_nodes_from_graph(graph)
    strategy = GossipStrategy(fanout=fanout)
    sim = Simulator(nodes=nodes, strategy=strategy, seed=42)
    message = Message("probe", source_id=1, created_at=0.0)
    return strategy.select_targets(
        nodes[0], message, sim, exclude_target_id=1
    )


class CapturingSimulator:
    instances: list["CapturingSimulator"] = []

    def __init__(self, nodes, strategy, controller=None, **kwargs):
        self.nodes = nodes
        self.strategy = strategy
        self.controller = controller
        self.metrics = self
        self.adaptive_trace_rows = []
        self.instances.append(self)

    def inject_message(self, **kwargs):
        pass

    def run(self):
        pass

    def summarize_message(self, *args, **kwargs):
        return {
            "delivery_ratio": 1.0,
            "propagation_delay": 0.0,
            "duplicates": 0,
            "total_forwards": 0,
        }

    def get_resource_metrics(self):
        return {}


def construction_probe(config_name: str, experiment) -> list[int | None]:
    cfg = load_yaml_config(str(ROOT / "configs" / config_name))
    cfg["runs_per_setting"] = 1
    cfg["strategies"] = ["gossip"]
    if cfg["experiment"] == "exp08":
        cfg["ch_overload_factor"] = [1.0]
    if cfg["experiment"] == "exp09":
        cfg["edge_probs"] = [0.08]

    CapturingSimulator.instances.clear()
    with (
        patch("run_batch.Simulator", CapturingSimulator),
        patch("run_batch.get_or_build_topology", return_value=nx.path_graph(7)),
    ):
        experiment(cfg)
    return [sim.strategy.fanout for sim in CapturingSimulator.instances]


def sender_handoff_probe() -> list[int]:
    graph = nx.star_graph(6)
    nodes = build_nodes_from_graph(graph)
    strategy = GossipStrategy()
    sim = Simulator(nodes=nodes, strategy=strategy, seed=42, base_delay=0.0, jitter=0.0)
    message = Message("handoff", source_id=1, created_at=0.0)
    sim.metrics.register_message("handoff", source_id=1, created_at=0.0)
    sent: list[int] = []
    sim.send_message = lambda src_id, dst_id, message, now: sent.append(dst_id)
    sim.handle_receive(now=0.0, dst_id=0, src_id=1, message=message, sent_at=0.0)
    return sent


def duplicate_probe() -> tuple[int, int]:
    nodes = build_nodes_from_graph(nx.path_graph(3))
    sim = Simulator(nodes=nodes, strategy=GossipStrategy(), seed=42)
    message = Message("duplicate", source_id=0, created_at=0.0)
    sim.metrics.register_message("duplicate", source_id=0, created_at=0.0)
    nodes[1].mark_seen("duplicate")
    sim.handle_receive(now=1.0, dst_id=1, src_id=0, message=message, sent_at=0.0)
    return nodes[1].stats.received_duplicate, sim.metrics.messages["duplicate"].duplicate_count


def main() -> None:
    fanout2 = direct_probe(2)
    fanout6 = direct_probe(6)
    normal = direct_probe(None)
    handoff = sender_handoff_probe()
    duplicate_counts = duplicate_probe()
    exp07_fanouts = construction_probe("exp07_fanout.yaml", exp07)
    exp08_fanouts = construction_probe("exp08_ch_bottleneck.yaml", exp08)
    exp09_fanouts = construction_probe("exp09_dense_topology.yaml", exp09)

    assert len(fanout2) == 2 and 1 not in fanout2
    assert len(fanout6) == 5 and 1 not in fanout6
    assert normal == [2, 3, 4, 5, 6]
    assert handoff == [2, 3, 4, 5, 6]
    assert duplicate_counts == (1, 1)
    assert exp07_fanouts == [2, 3, 4, 5, 6]
    assert exp08_fanouts == [None]
    assert exp09_fanouts == [None]

    print("Exp07 configured fanouts:", exp07_fanouts)
    print("Exp07 fanout=2 targets:", fanout2)
    print("Exp07 fanout=6 targets:", fanout6)
    print("Exp08 constructed fanout:", exp08_fanouts)
    print("Exp09 constructed fanout:", exp09_fanouts)
    print("Normal Gossip targets:", normal)
    print("Simulator sender-handoff targets:", handoff)
    print("Duplicate accounting (node, metrics):", duplicate_counts)
    print("S0 Gossip targeted probes: PASS")


if __name__ == "__main__":
    main()
