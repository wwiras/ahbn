from __future__ import annotations

"""
Exp13: RO2 Scalability Sensitivity Experiment

Purpose
-------
This controlled Python simulation directly addresses the panel comment about
why the earlier message-duplication analysis used 1000 nodes. It repeats the
same type of static-protocol comparison across multiple network sizes.

Protocols
---------
- rns : Random Neighbor Selection / gossip-style forwarding.
- km  : k-Means-inspired clustered forwarding.
- ac  : Acyclic tree dissemination baseline.

This script is intentionally independent from run_batch.py so it does not
change Exp07--Exp12. SBA/AHBN can be added later as another strategy.
"""

import argparse
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import networkx as nx
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.message import Message
from ahbn.node import Node
from ahbn.simulator import Simulator
from ahbn.strategies.base import ForwardingStrategy
from ahbn.strategies.gossip import GossipStrategy
from ahbn.topology import build_nodes_from_graph, get_or_build_topology
from ahbn.utils import current_timestamp


# ---------------------------------------------------------------------------
# Helper clustering utilities
# ---------------------------------------------------------------------------


def make_er_connected(num_nodes: int, kavg: int, seed: int, use_cache: bool = True) -> nx.Graph:
    """Build an ER graph with expected average degree approximately kavg."""
    p = min(1.0, float(kavg) / max(1, num_nodes - 1))
    return get_or_build_topology(
        topology_type="er",
        num_nodes=num_nodes,
        seed=seed,
        use_cache=use_cache,
        edge_prob=round(p, 6),
        ba_m=None,
    )


def choose_num_clusters(num_nodes: int, target_cluster_size: int = 25) -> int:
    """
    Keep cluster size approximately stable as n grows.
    This is important because the panel concern is about node scale, not about
    silently changing cluster density.
    """
    return max(2, int(round(num_nodes / target_cluster_size)))


def deterministic_positions(node_ids: Iterable[int], seed: int) -> Dict[int, Tuple[float, float]]:
    """Simple 2D node features for k-means clustering; fast and reproducible."""
    rng = random.Random(seed)
    return {nid: (rng.random(), rng.random()) for nid in node_ids}


def kmeans_assignments(
    node_ids: List[int],
    positions: Dict[int, Tuple[float, float]],
    k: int,
    seed: int,
    iterations: int = 20,
) -> Dict[int, int]:
    """Small dependency-free k-means implementation for 2D points."""
    rng = random.Random(seed)
    k = max(1, min(k, len(node_ids)))
    centroids = [positions[nid] for nid in rng.sample(node_ids, k)]

    assignments: Dict[int, int] = {}
    for _ in range(iterations):
        changed = False
        buckets: List[List[int]] = [[] for _ in range(k)]

        for nid in node_ids:
            x, y = positions[nid]
            cid = min(
                range(k),
                key=lambda c: (x - centroids[c][0]) ** 2 + (y - centroids[c][1]) ** 2,
            )
            if assignments.get(nid) != cid:
                changed = True
            assignments[nid] = cid
            buckets[cid].append(nid)

        for cid, members in enumerate(buckets):
            if members:
                centroids[cid] = (
                    sum(positions[n][0] for n in members) / len(members),
                    sum(positions[n][1] for n in members) / len(members),
                )

        if not changed:
            break

    return assignments


def apply_cluster_metadata(nodes: Dict[int, Node], assignments: Dict[int, int]) -> Dict[int, List[int]]:
    clusters: Dict[int, List[int]] = {}
    for nid, cid in assignments.items():
        clusters.setdefault(cid, []).append(nid)
        nodes[nid].cluster_id = cid

    for cid, members in clusters.items():
        # Degree-aware head selection within each cluster.
        head = max(members, key=lambda n: (len(nodes[n].neighbors), -n))
        nodes[head].is_cluster_head = True

    return clusters


# ---------------------------------------------------------------------------
# Exp13 protocol strategies
# ---------------------------------------------------------------------------


class KMeansForwardingStrategy(ForwardingStrategy):
    """
    KM baseline: clustered but still cyclic within clusters.

    Interpretation:
    - Each normal node forwards to its cluster head plus a few same-cluster peers.
    - Each cluster head forwards to all members in its cluster and to neighboring
      cluster heads in a ring.

    This produces less random flooding than RNS, but still allows duplicate paths
    inside clusters, matching the RO2 interpretation of KM as an intermediate
    cyclic structured baseline.
    """

    def __init__(self, clusters: Dict[int, List[int]], fanout: int = 25, seed: int = 42) -> None:
        self.clusters = clusters
        self.fanout = fanout
        self.rng = random.Random(seed)
        self.cluster_heads: Dict[int, int] = {}
        self.head_ring: Dict[int, List[int]] = {}

    def bind_nodes(self, nodes: Dict[int, Node]) -> None:
        for cid, members in self.clusters.items():
            heads = [n for n in members if nodes[n].is_cluster_head]
            self.cluster_heads[cid] = heads[0] if heads else min(members)

        ordered_heads = [self.cluster_heads[cid] for cid in sorted(self.cluster_heads)]
        if len(ordered_heads) <= 1:
            return
        for i, h in enumerate(ordered_heads):
            self.head_ring[h] = [
                ordered_heads[(i - 1) % len(ordered_heads)],
                ordered_heads[(i + 1) % len(ordered_heads)],
            ]

    def _sample(self, simulator: Simulator, candidates: List[int], k: int) -> List[int]:
        candidates = list(dict.fromkeys(candidates))
        if k <= 0 or not candidates:
            return []
        if len(candidates) <= k:
            return candidates
        return simulator.rng.sample(candidates, k)

    def select_targets(self, node: Node, message: Message, simulator: Simulator) -> List[int]:
        cid = node.cluster_id
        if cid is None:
            return []

        members = [m for m in self.clusters[cid] if m != node.node_id]
        head = self.cluster_heads[cid]

        if node.is_cluster_head:
            # Head disseminates inside cluster and links cluster-to-cluster.
            targets = members[:] + self.head_ring.get(node.node_id, [])
            return [t for t in dict.fromkeys(targets) if t != node.node_id]

        same_cluster_neighbors = [
            n for n in node.neighbors
            if n != node.node_id and simulator.nodes[n].cluster_id == cid
        ]
        targets: List[int] = []
        if head != node.node_id:
            targets.append(head)
        targets.extend(self._sample(simulator, same_cluster_neighbors, max(0, self.fanout - 1)))
        return [t for t in dict.fromkeys(targets) if t != node.node_id]


class AcyclicTreeStrategy(ForwardingStrategy):
    """
    AC baseline: acyclic dissemination over a BFS spanning tree.

    The strategy forwards only from parent to children, so duplicate messages are
    expected to be near zero. The trade-off is that propagation delay follows the
    height/depth of the tree rather than many parallel cyclic paths.
    """

    def __init__(self, graph: nx.Graph, source_id: int) -> None:
        tree = nx.bfs_tree(graph, source=source_id)
        self.children: Dict[int, List[int]] = {n: [] for n in graph.nodes()}
        for parent, child in tree.edges():
            self.children[int(parent)].append(int(child))

    def select_targets(self, node: Node, message: Message, simulator: Simulator) -> List[int]:
        return self.children.get(node.node_id, [])


# ---------------------------------------------------------------------------
# Running and plotting
# ---------------------------------------------------------------------------


@dataclass
class Exp13Row:
    experiment: str
    strategy: str
    seed: int
    num_nodes: int
    kavg: int
    num_clusters: int | None
    delivery_ratio: float
    propagation_delay: float | None
    duplicates: int
    duplicate_ratio: float
    total_forwards: int


def run_one_exp13(
    strategy_name: str,
    num_nodes: int,
    kavg: int,
    seed: int,
    base_delay: float,
    jitter: float,
    message_source: int,
    use_topology_cache: bool,
    target_cluster_size: int,
) -> Exp13Row:
    graph = make_er_connected(num_nodes=num_nodes, kavg=kavg, seed=seed, use_cache=use_topology_cache)
    nodes = build_nodes_from_graph(graph)
    actual_n = len(nodes)
    source_id = message_source if message_source in nodes else min(nodes)

    num_clusters: int | None = None

    if strategy_name == "rns":
        strategy = GossipStrategy(fanout=kavg)

    elif strategy_name == "km":
        num_clusters = choose_num_clusters(actual_n, target_cluster_size=target_cluster_size)
        positions = deterministic_positions(sorted(nodes), seed=seed)
        assignments = kmeans_assignments(sorted(nodes), positions, k=num_clusters, seed=seed)
        clusters = apply_cluster_metadata(nodes, assignments)
        strategy = KMeansForwardingStrategy(clusters=clusters, fanout=kavg, seed=seed)
        strategy.bind_nodes(nodes)

    elif strategy_name == "ac":
        strategy = AcyclicTreeStrategy(graph=graph, source_id=source_id)

    else:
        raise ValueError(f"Unknown strategy: {strategy_name}")

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=None,
        controller=None,
        experiment_name="exp13",
        strategy_name=strategy_name,
        scenario_tag=f"n{actual_n}",
    )
    sim.inject_message(source_id=source_id, message_id="m1")
    sim.run(until=10000.0)

    summary = sim.metrics.summarize_message("m1", total_nodes=actual_n)
    total_forwards = int(summary["total_forwards"])
    duplicates = int(summary["duplicates"])
    duplicate_ratio = duplicates / total_forwards if total_forwards > 0 else 0.0

    return Exp13Row(
        experiment="exp13",
        strategy=strategy_name,
        seed=seed,
        num_nodes=actual_n,
        kavg=kavg,
        num_clusters=num_clusters,
        delivery_ratio=float(summary["delivery_ratio"]),
        propagation_delay=summary["propagation_delay"],
        duplicates=duplicates,
        duplicate_ratio=duplicate_ratio,
        total_forwards=total_forwards,
    )


def run_exp13(args: argparse.Namespace) -> pd.DataFrame:
    rows: List[Exp13Row] = []
    strategies = [s.strip().lower() for s in args.strategies.split(",") if s.strip()]

    for n in args.nodes:
        for run_idx in range(args.runs):
            seed = args.seed + run_idx
            for strategy in strategies:
                print(f"Running exp13 strategy={strategy} n={n} seed={seed}", flush=True)
                rows.append(
                    run_one_exp13(
                        strategy_name=strategy,
                        num_nodes=n,
                        kavg=args.kavg,
                        seed=seed,
                        base_delay=args.base_delay,
                        jitter=args.jitter,
                        message_source=args.message_source,
                        use_topology_cache=not args.no_cache,
                        target_cluster_size=args.target_cluster_size,
                    )
                )

    df = pd.DataFrame([r.__dict__ for r in rows])
    return df


def save_outputs(df: pd.DataFrame, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    ts = current_timestamp()

    csv_path = output_dir / f"exp13_scalability_results_{ts}.csv"

    summary_path = output_dir / f"exp13_scalability_summary_{ts}.csv"

    panel_table_path = output_dir / f"exp13_panel_table_{ts}.csv"

    # plot_path = plot_dir / f"exp13_scalability_plot_{ts}.png"
    plot_path = plot_dir / f"exp13_node_scalability_delay_duplication_{ts}.png"

    df.to_csv(csv_path, index=False)

    # ==========================================================
    # Summary with mean and std
    # ==========================================================

    summary = (
        df.groupby(["num_nodes", "strategy"], as_index=False)
        .agg(
            propagation_delay_mean=("propagation_delay", "mean"),
            propagation_delay_std=("propagation_delay", "std"),

            duplicate_ratio_mean=("duplicate_ratio", "mean"),
            duplicate_ratio_std=("duplicate_ratio", "std"),
        )
        .sort_values(["strategy", "num_nodes"])
    )

    summary.to_csv(summary_path, index=False)

    # ==========================================================
    # Panel-ready table
    # ==========================================================

    panel_table = summary.copy()

    panel_table["Delay (mean±std)"] = (
        panel_table["propagation_delay_mean"].round(2).astype(str)
        + " ± "
        + panel_table["propagation_delay_std"].fillna(0).round(2).astype(str)
    )

    panel_table["Duplicate Ratio (mean±std)"] = (
        panel_table["duplicate_ratio_mean"].round(3).astype(str)
        + " ± "
        + panel_table["duplicate_ratio_std"].fillna(0).round(3).astype(str)
    )

    panel_table = panel_table[
        [
            "num_nodes",
            "strategy",
            "Delay (mean±std)",
            "Duplicate Ratio (mean±std)",
        ]
    ]

    panel_table.to_csv(panel_table_path, index=False)

    print("\n")
    print("=" * 80)
    print("EXP13 PANEL SUMMARY")
    print("=" * 80)
    print(panel_table.to_string(index=False))
    print("=" * 80)

    plot_summary(summary, plot_path)

    return (
        csv_path,
        summary_path,
        panel_table_path,
        plot_path,
    )

def plot_summary(summary: pd.DataFrame, plot_path: Path) -> None:

    strategies = list(summary["strategy"].unique())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # ======================================================
    # Left plot : Duplicate Ratio
    # ======================================================

    ax = axes[0]

    for strategy in strategies:

        sub = (
            summary[summary["strategy"] == strategy]
            .sort_values("num_nodes")
        )

        ax.errorbar(
            sub["num_nodes"],
            sub["duplicate_ratio_mean"],
            yerr=sub["duplicate_ratio_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=strategy.upper(),
        )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Duplicate Ratio")
    ax.set_title("Message Duplication")
    ax.grid(True, linestyle="--", alpha=0.5)

    # ======================================================
    # Right plot : Propagation Delay
    # ======================================================

    ax = axes[1]

    for strategy in strategies:

        sub = (
            summary[summary["strategy"] == strategy]
            .sort_values("num_nodes")
        )

        ax.errorbar(
            sub["num_nodes"],
            sub["propagation_delay_mean"],
            yerr=sub["propagation_delay_std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=strategy.upper(),
        )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Propagation Delay")
    ax.set_title("Propagation Performance")
    ax.grid(True, linestyle="--", alpha=0.5)

    handles, labels = axes[0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=False,
    )

    fig.suptitle(
        "Impact of Network Size on Dissemination Scalability",
        fontsize=14,
        fontweight="bold",
    )

    fig.tight_layout(rect=[0, 0, 1, 0.93])

    fig.savefig(
        plot_path,
        dpi=300,
        bbox_inches="tight",
    )

    plt.close(fig)

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Exp13 RO2 scalability sensitivity experiment.")
    parser.add_argument("--nodes", nargs="+", type=int, default=[100, 500, 1000, 2000])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kavg", type=int, default=25)
    parser.add_argument("--target-cluster-size", type=int, default=25)
    parser.add_argument("--strategies", type=str, default="rns,km,ac")
    parser.add_argument("--base-delay", type=float, default=1.0)
    parser.add_argument("--jitter", type=float, default=0.2)
    parser.add_argument("--message-source", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/csv"))
    parser.add_argument("--no-cache", action="store_true")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    df = run_exp13(args)

    csv_path, summary_path, panel_table_path, plot_path = save_outputs(
        df,
        args.output_dir,
    )

    print(f"Saved raw results: {csv_path}")
    print(f"Saved summary:     {summary_path}")
    print(f"Saved panel table: {panel_table_path}")
    print(f"Saved plot:        {plot_path}")

if __name__ == "__main__":
    main()
