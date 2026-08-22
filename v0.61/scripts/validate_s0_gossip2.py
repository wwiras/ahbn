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


def probe_graph(eligible_count: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(eligible_count + 2))
    graph.add_edges_from((1, peer) for peer in [0, *range(2, eligible_count + 2)])
    return graph


class ForwardCaptureSimulator(Simulator):
    decisions: list[tuple[int | None, list[int]]] = []

    def inject_message(self, source_id: int, message_id: str) -> None:
        message = Message(message_id, source_id=0, created_at=0.0)
        self.metrics.register_message(message_id, source_id=0, created_at=0.0)
        targets: list[int] = []
        self.send_message = lambda src_id, dst_id, message, now: targets.append(dst_id)
        self.handle_receive(
            now=0.0,
            dst_id=1,
            src_id=0,
            message=message,
            sent_at=0.0,
        )
        self.decisions.append((self.strategy.fanout, targets))

    def run(self) -> None:
        pass


def experiment_probe(name: str, eligible_count: int, fanout: int | None = None) -> list[int]:
    cfg = load_yaml_config(str(ROOT / "configs" / {
        "exp07": "exp07_fanout.yaml",
        "exp08": "exp08_ch_bottleneck.yaml",
        "exp09": "exp09_dense_topology.yaml",
    }[name]))
    cfg["runs_per_setting"] = 1
    cfg["strategies"] = ["gossip"]
    if name == "exp07":
        cfg["fanouts"] = [fanout]
        runner = exp07
    elif name == "exp08":
        cfg["ch_overload_factor"] = [1.0]
        runner = exp08
    else:
        cfg["edge_probs"] = [0.08]
        runner = exp09

    ForwardCaptureSimulator.decisions.clear()
    with (
        patch("run_batch.Simulator", ForwardCaptureSimulator),
        patch("run_batch.get_or_build_topology", return_value=probe_graph(eligible_count)),
    ):
        runner(cfg)
    configured, targets = ForwardCaptureSimulator.decisions[0]
    assert configured == fanout
    return targets


def degree_probe(name: str, eligible_count: int) -> int:
    cfg = load_yaml_config(str(ROOT / "configs" / {
        "exp08": "exp08_ch_bottleneck.yaml",
        "exp09": "exp09_dense_topology.yaml",
    }[name]))
    ForwardCaptureSimulator.decisions.clear()
    with (
        patch("run_batch.Simulator", ForwardCaptureSimulator),
        patch("run_batch.get_or_build_topology", return_value=probe_graph(eligible_count)),
    ):
        run_single(
            cfg=cfg,
            strategy_name="gossip",
            seed=42,
            topology_type=cfg["topology_type"],
            num_nodes=eligible_count + 2,
            use_topology_cache=False,
            base_delay=1.0,
            jitter=0.0,
            message_source=0,
            fanout=None,
            edge_prob=0.08 if name == "exp09" else None,
            ba_m=3 if name == "exp08" else None,
        )
    configured, targets = ForwardCaptureSimulator.decisions[0]
    assert configured is None and 0 not in targets
    return len(targets)


def event_queue_probe() -> tuple[list[int], int]:
    nodes = build_nodes_from_graph(probe_graph(7))
    sim = Simulator(nodes=nodes, strategy=GossipStrategy(), seed=42)
    message = Message("events", source_id=0, created_at=0.0)
    sim.metrics.register_message("events", source_id=0, created_at=0.0)
    sim.handle_receive(0.0, dst_id=1, src_id=0, message=message, sent_at=0.0)
    destinations = sorted(event.payload["dst_id"] for event in sim.queue)
    return destinations, len(sim.queue)


def main() -> None:
    cases = {
        ("Exp07 f=2", 7): experiment_probe("exp07", 7, 2),
        ("Exp07 f=6", 7): experiment_probe("exp07", 7, 6),
        ("Exp08", 7): experiment_probe("exp08", 7),
        ("Exp09", 7): experiment_probe("exp09", 7),
        ("Exp08", 9): experiment_probe("exp08", 9),
        ("Exp09", 9): experiment_probe("exp09", 9),
    }
    expected = [2, 6, 7, 7, 9, 9]
    assert [len(targets) for targets in cases.values()] == expected
    assert all(0 not in targets for targets in cases.values())

    degree_counts = {
        name: [degree_probe(name, degree) for degree in range(1, 10)]
        for name in ("exp08", "exp09")
    }
    assert degree_counts["exp08"] == list(range(1, 10))
    assert degree_counts["exp09"] == list(range(1, 10))

    destinations, event_count = event_queue_probe()
    assert destinations == list(range(2, 9)) and event_count == 7

    print("Experiment   Eligible   Configured fanout   Actual count   Actual targets")
    for (label, eligible), targets in cases.items():
        configured = label.split("=")[-1] if "=" in label else "NONE"
        print(f"{label:<12} {eligible:>8}   {configured:^17}   {len(targets):>12}   {sorted(targets)}")
    print("Exp08 degree 1..9 target counts:", degree_counts["exp08"])
    print("Exp09 degree 1..9 target counts:", degree_counts["exp09"])
    print("One decision selected targets:", destinations)
    print("Receive events scheduled from that decision:", event_count)
    print("S0 Gossip additional targeted probes: PASS")


if __name__ == "__main__":
    main()
