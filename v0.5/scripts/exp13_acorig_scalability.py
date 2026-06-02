from __future__ import annotations

"""
Exp13-ACORIG: RO2 Scalability Sensitivity with Original AC-style MST Baseline

Protocols:
- rns     : Random Neighbor Selection / gossip-style forwarding
- km      : k-Means-inspired clustered forwarding
- ac_orig : Agglomerative-style clustering + MST overlay + parent->children dissemination

Purpose:
This script is separate from the existing Exp13 script so it does not interfere
with the previous AC/BFS-tree implementation.
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
# Topology and clustering helpers
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
    """Keep approximate cluster size stable as n grows."""
    return max(2, int(round(num_nodes / target_cluster_size)))


def deterministic_positions(node_ids: Iterable[int], seed: int) -> Dict[int, Tuple[float, float]]:
    """Simple 2D node features for k-means clustering."""
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
        head = max(members, key=lambda n: (len(nodes[n].neighbors), -n))
        nodes[head].is_cluster_head = True

    return clusters


# ---------------------------------------------------------------------------
# AC_ORIG scalable approximation
# ---------------------------------------------------------------------------

def greedy_agglomerative_clusters_by_graph_distance(
    graph: nx.Graph,
    num_clusters: int,
) -> Dict[int, List[int]]:
    """
    Scalable approximation of original AC behavior.

    Instead of full O(n^3) complete-linkage agglomerative clustering, this method
    repeatedly merges connected components using graph edges ordered by distance.
    It preserves the key behavior needed for Exp13:
    - forms local structural clusters from graph proximity
    - cluster count is controlled
    - suitable for 100--2000 nodes
    """

    components: Dict[int, List[int]] = {int(n): [int(n)] for n in graph.nodes()}
    parent: Dict[int, int] = {int(n): int(n) for n in graph.nodes()}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> bool:
        ra, rb = find(a), find(b)
        if ra == rb:
            return False

        if len(components[ra]) < len(components[rb]):
            ra, rb = rb, ra

        parent[rb] = ra
        components[ra].extend(components[rb])
        del components[rb]
        return True

    # ER graph edges usually do not have weight, so use weight=1.0 by default.
    edges = [
        (int(u), int(v), float(data.get("weight", 1.0)))
        for u, v, data in graph.edges(data=True)
    ]

    # Stable sorting: lower weight first, then node ids.
    edges.sort(key=lambda x: (x[2], min(x[0], x[1]), max(x[0], x[1])))

    for u, v, _w in edges:
        if len(components) <= num_clusters:
            break
        union(u, v)

    # If graph structure cannot merge enough, merge smallest components deterministically.
    while len(components) > num_clusters:
        roots = sorted(components, key=lambda r: (len(components[r]), r))
        a, b = roots[0], roots[1]
        union(a, b)

    return {
        cid: sorted(members)
        for cid, members in enumerate(components.values())
    }


def build_mst_edges_for_nodes(graph: nx.Graph, members: List[int]) -> List[Tuple[int, int]]:
    """
    Build MST edges inside a cluster. If the induced subgraph is disconnected,
    connect missing members using shortest paths from the original graph where possible.
    """
    if len(members) <= 1:
        return []

    sub = graph.subgraph(members).copy()

    if sub.number_of_edges() == 0:
        # fallback chain if no internal edges exist
        return [(members[i], members[i + 1]) for i in range(len(members) - 1)]

    mst_edges: List[Tuple[int, int]] = []

    # MST per connected component.
    for comp in nx.connected_components(sub):
        comp_sub = sub.subgraph(comp).copy()
        if comp_sub.number_of_nodes() <= 1:
            continue
        mst = nx.minimum_spanning_tree(comp_sub, weight="weight")
        mst_edges.extend((int(u), int(v)) for u, v in mst.edges())

    return mst_edges


def choose_cluster_root(graph: nx.Graph, members: List[int]) -> int:
    """Degree-aware cluster root selection."""
    return max(members, key=lambda n: (graph.degree[n], -n))


def build_ac_orig_overlay_tree(
    graph: nx.Graph,
    num_clusters: int,
    source_id: int,
) -> nx.Graph:
    """
    Build AC_ORIG overlay:
    1. Agglomerative-style clusters
    2. MST inside each cluster
    3. MST among cluster roots
    4. Return one connected acyclic overlay graph
    """
    clusters = greedy_agglomerative_clusters_by_graph_distance(
        graph=graph,
        num_clusters=num_clusters,
    )

    overlay = nx.Graph()
    overlay.add_nodes_from(int(n) for n in graph.nodes())

    cluster_roots: List[int] = []

    for _cid, members in clusters.items():
        root = choose_cluster_root(graph, members)
        cluster_roots.append(root)

        internal_edges = build_mst_edges_for_nodes(graph, members)
        overlay.add_edges_from(internal_edges)

    # Connect cluster roots using shortest-path distance on the original graph.
    if len(cluster_roots) > 1:
        root_complete = nx.Graph()
        root_complete.add_nodes_from(cluster_roots)

        for i, a in enumerate(cluster_roots):
            lengths = nx.single_source_shortest_path_length(graph, a)
            for b in cluster_roots[i + 1:]:
                dist = lengths.get(b, float("inf"))
                if dist != float("inf"):
                    root_complete.add_edge(a, b, weight=float(dist))

        if root_complete.number_of_edges() > 0:
            root_mst = nx.minimum_spanning_tree(root_complete, weight="weight")
            overlay.add_edges_from((int(u), int(v)) for u, v in root_mst.edges())

    # Final safety: if overlay is disconnected, connect components using graph shortest paths.
    while not nx.is_connected(overlay):
        comps = [list(c) for c in nx.connected_components(overlay)]
        base = comps[0]
        best_pair = None
        best_dist = float("inf")

        for a in base:
            lengths = nx.single_source_shortest_path_length(graph, a)
            for comp in comps[1:]:
                for b in comp:
                    d = lengths.get(b, float("inf"))
                    if d < best_dist:
                        best_dist = d
                        best_pair = (a, b)

        if best_pair is None:
            # Last-resort deterministic bridge.
            overlay.add_edge(comps[0][0], comps[1][0])
        else:
            overlay.add_edge(int(best_pair[0]), int(best_pair[1]))

    # Convert to one tree if accidental cycles appear.
    if overlay.number_of_edges() > overlay.number_of_nodes() - 1:
        overlay = nx.minimum_spanning_tree(overlay)

    return overlay


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

class KMeansForwardingStrategy(ForwardingStrategy):
    """
    KM baseline:
    - normal nodes forward to cluster head + same-cluster peers
    - cluster heads forward within cluster + to neighboring cluster heads
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
            targets = members[:] + self.head_ring.get(node.node_id, [])
            return [t for t in dict.fromkeys(targets) if t != node.node_id]

        same_cluster_neighbors = [
            n for n in node.neighbors
            if n != node.node_id and simulator.nodes[n].cluster_id == cid
        ]

        targets: List[int] = []

        if head != node.node_id:
            targets.append(head)

        targets.extend(
            self._sample(
                simulator,
                same_cluster_neighbors,
                max(0, self.fanout - 1),
            )
        )

        return [t for t in dict.fromkeys(targets) if t != node.node_id]


class AgglomerativeMSTStrategy(ForwardingStrategy):
    """
    AC_ORIG baseline:
    Agglomerative-style clustering + MST overlay, then parent->children forwarding.

    This intentionally creates a structured acyclic dissemination path:
    - duplicate ratio should be near zero
    - delay may increase due to hierarchical MST paths
    """

    def __init__(self, graph: nx.Graph, source_id: int, num_clusters: int) -> None:
        overlay_tree = build_ac_orig_overlay_tree(
            graph=graph,
            num_clusters=num_clusters,
            source_id=source_id,
        )

        directed_tree = nx.bfs_tree(overlay_tree, source=source_id)

        self.children: Dict[int, List[int]] = {int(n): [] for n in overlay_tree.nodes()}

        for parent, child in directed_tree.edges():
            self.children[int(parent)].append(int(child))

    def select_targets(self, node: Node, message: Message, simulator: Simulator) -> List[int]:
        return self.children.get(node.node_id, [])


# ---------------------------------------------------------------------------
# Experiment logic
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

    graph = make_er_connected(
        num_nodes=num_nodes,
        kavg=kavg,
        seed=seed,
        use_cache=use_topology_cache,
    )

    nodes = build_nodes_from_graph(graph)
    actual_n = len(nodes)
    source_id = message_source if message_source in nodes else min(nodes)
    num_clusters: int | None = None

    if strategy_name == "rns":
        strategy = GossipStrategy(fanout=kavg)

    elif strategy_name == "km":
        num_clusters = choose_num_clusters(
            actual_n,
            target_cluster_size=target_cluster_size,
        )

        positions = deterministic_positions(sorted(nodes), seed=seed)

        assignments = kmeans_assignments(
            sorted(nodes),
            positions,
            k=num_clusters,
            seed=seed,
        )

        clusters = apply_cluster_metadata(nodes, assignments)

        strategy = KMeansForwardingStrategy(
            clusters=clusters,
            fanout=kavg,
            seed=seed,
        )

        strategy.bind_nodes(nodes)

    elif strategy_name == "ac_orig":
        num_clusters = choose_num_clusters(
            actual_n,
            target_cluster_size=target_cluster_size,
        )

        strategy = AgglomerativeMSTStrategy(
            graph=graph,
            source_id=source_id,
            num_clusters=num_clusters,
        )

    else:
        raise ValueError(
            f"Unknown strategy: {strategy_name}. "
            f"Use one of: rns, km, ac_orig"
        )

    sim = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=seed,
        base_delay=base_delay,
        jitter=jitter,
        cluster_manager=None,
        controller=None,
        experiment_name="exp13_acorig",
        strategy_name=strategy_name,
        scenario_tag=f"n{actual_n}",
    )

    sim.inject_message(
        source_id=source_id,
        message_id="m1",
    )

    sim.run(until=10000.0)

    summary = sim.metrics.summarize_message(
        "m1",
        total_nodes=actual_n,
    )

    total_forwards = int(summary["total_forwards"])
    duplicates = int(summary["duplicates"])
    duplicate_ratio = duplicates / total_forwards if total_forwards > 0 else 0.0

    return Exp13Row(
        experiment="exp13_acorig",
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
                print(
                    f"Running exp13_acorig strategy={strategy} "
                    f"n={n} seed={seed}",
                    flush=True,
                )

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

    return pd.DataFrame([r.__dict__ for r in rows])


# ---------------------------------------------------------------------------
# Output and plotting
# ---------------------------------------------------------------------------

def save_outputs(df: pd.DataFrame, output_dir: Path):

    output_dir.mkdir(parents=True, exist_ok=True)

    plot_dir = Path("outputs/plots")
    plot_dir.mkdir(parents=True, exist_ok=True)

    ts = current_timestamp()

    csv_path = output_dir / f"exp13_acorig_scalability_results_{ts}.csv"
    summary_path = output_dir / f"exp13_acorig_scalability_summary_{ts}.csv"
    panel_table_path = output_dir / f"exp13_acorig_panel_table_{ts}.csv"
    plot_path = plot_dir / f"exp13_acorig_node_scalability_delay_duplication_{ts}.png"

    df.to_csv(csv_path, index=False)

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
    print("EXP13-ACORIG PANEL SUMMARY")
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

    style_map = {
        "rns": {"marker": "o", "linestyle": "-"},
        "km": {"marker": "s", "linestyle": "-"},
        "ac_orig": {"marker": "^", "linestyle": "--"},
    }

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 5),
    )

    # Duplicate ratio
    ax = axes[0]

    for strategy in strategies:
        sub = summary[summary["strategy"] == strategy].sort_values("num_nodes")
        style = style_map.get(strategy, {"marker": "o", "linestyle": "-"})

        ax.errorbar(
            sub["num_nodes"],
            sub["duplicate_ratio_mean"],
            yerr=sub["duplicate_ratio_std"],
            marker=style["marker"],
            linestyle=style["linestyle"],
            linewidth=2,
            capsize=4,
            label=strategy.upper(),
        )

    ax.set_xlabel("Number of Nodes")
    ax.set_ylabel("Duplicate Ratio")
    ax.set_title("Message Duplication")
    ax.grid(True, linestyle="--", alpha=0.5)

    # Propagation delay
    ax = axes[1]

    for strategy in strategies:
        sub = summary[summary["strategy"] == strategy].sort_values("num_nodes")
        style = style_map.get(strategy, {"marker": "o", "linestyle": "-"})

        ax.errorbar(
            sub["num_nodes"],
            sub["propagation_delay_mean"],
            yerr=sub["propagation_delay_std"],
            marker=style["marker"],
            linestyle=style["linestyle"],
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


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Exp13-ACORIG RO2 scalability sensitivity experiment."
    )

    parser.add_argument("--nodes", nargs="+", type=int, default=[100, 500, 1000, 2000])
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--kavg", type=int, default=25)
    parser.add_argument("--target-cluster-size", type=int, default=25)
    parser.add_argument("--strategies", type=str, default="rns,km,ac_orig")
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