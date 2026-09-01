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
from run_batch import exp07, exp08, exp09  # noqa: E402


def probe_graph(eligible_count: int) -> nx.Graph:
    graph = nx.Graph()
    graph.add_nodes_from(range(eligible_count + 2))
    graph.add_edges_from((1, peer) for peer in [0, *range(2, eligible_count + 2)])
    return graph


class CaptureSimulator(Simulator):
    decisions: list[tuple[int | None, list[int], object, object, int]] = []

    def inject_message(self, source_id: int, message_id: str) -> None:
        message = Message(message_id, source_id=0, created_at=0.0)
        self.metrics.register_message(message_id, source_id=0, created_at=0.0)
        targets: list[int] = []
        self.send_message = lambda src_id, dst_id, message, now: targets.append(dst_id)
        self.handle_receive(0.0, dst_id=1, src_id=0, message=message, sent_at=0.0)
        self.decisions.append(
            (self.strategy.fanout, targets, self.controller, self.cluster_manager, len(self.adaptive_trace_rows))
        )

    def run(self) -> None:
        pass


def experiment_probe(name: str, eligible_count: int, fanout: int | None = None):
    config_names = {
        "exp07": "exp07_fanout.yaml",
        "exp08": "exp08_ch_bottleneck.yaml",
        "exp09": "exp09_dense_topology.yaml",
    }
    cfg = load_yaml_config(str(ROOT / "configs" / config_names[name]))
    cfg["runs_per_setting"] = 1
    cfg["strategies"] = ["gossip"]
    if name == "exp07":
        cfg["fanouts"] = [fanout]
        runner = exp07
    elif name == "exp08":
        cfg["ch_overload_factor"] = [1.0]
        runner = exp08
    else:
        cfg["edge_probs"] = [0.04 if eligible_count == 3 else 0.12]
        runner = exp09

    CaptureSimulator.decisions.clear()
    with (
        patch("run_batch.Simulator", CaptureSimulator),
        patch("run_batch.get_or_build_topology", return_value=probe_graph(eligible_count)),
    ):
        runner(cfg)
    return CaptureSimulator.decisions[0]


def main() -> None:
    print("Exp07: seven eligible peers, sender=0")
    for k in range(2, 7):
        configured, targets, controller, clusters, updates = experiment_probe("exp07", 7, k)
        assert configured == k
        assert len(targets) == k
        assert 0 not in targets
        assert controller is None and clusters is None and updates == 0
        print(f"k={k}: eligible=7 selected={len(targets)} targets={sorted(targets)} sender_excluded=PASS")

    configured, targets, *_ = experiment_probe("exp07", 3, 6)
    assert configured == 6 and sorted(targets) == [2, 3, 4]
    print("k=6 shortfall: eligible=3 selected=3 targets=[2, 3, 4] PASS")

    for name in ("exp08", "exp09"):
        configured, targets, controller, clusters, updates = experiment_probe(name, 7)
        assert configured is None
        assert sorted(targets) == list(range(2, 9))
        assert 0 not in targets
        assert controller is None and clusters is None and updates == 0
        print(f"{name}: eligible=7 selected=7 fanout=None sender_excluded=PASS controller=None updates=0")

    low = experiment_probe("exp09", 3)[1]
    high = experiment_probe("exp09", 9)[1]
    assert len(low) == 3 and len(high) == 9
    assert 0 not in low and 0 not in high
    print("Exp09 density: low eligible/selected=3/3 high eligible/selected=9/9 PASS")
    print("S1 Gossip small regression tests: PASS")


if __name__ == "__main__":
    main()
