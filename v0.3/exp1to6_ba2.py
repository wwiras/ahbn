from __future__ import annotations

import csv
import os
import sys
import random
from datetime import datetime
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional

import networkx as nx


# -----------------------------------------------------------------------------
# Global experiment controls
# -----------------------------------------------------------------------------

RUNS = 100
BASE_SEED = 42


# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------

@dataclass
class SimulationResult:
    protocol: str
    node_count: int
    informed_count: int
    reachable_count: int
    delivery_ratio: float
    rounds: int
    transmissions: int
    duplicates: int
    duplicate_ratio: float
    propagation_efficiency: float
    avg_path_length: float
    avg_degree: float
    clustering_coeff: float


# -----------------------------------------------------------------------------
# Core simulator
# -----------------------------------------------------------------------------

class BASimulator:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)

    def build_ba_graph(self, node_count: int, ba_m: int) -> nx.Graph:
        ba_m = max(1, min(ba_m, node_count - 1))
        return nx.barabasi_albert_graph(node_count, ba_m, seed=self.seed)

    def derive_clusters_from_graph(
        self,
        g: nx.Graph,
        ch_count: int,
    ) -> Tuple[List[int], Dict[int, int]]:
        node_count = g.number_of_nodes()
        ch_count = max(1, min(ch_count, node_count))

        degree_sorted = sorted(g.degree(), key=lambda x: (-x[1], x[0]))
        cluster_heads = [node for node, _ in degree_sorted[:ch_count]]

        node_to_ch: Dict[int, int] = {}
        sp_maps = {
            ch: nx.single_source_shortest_path_length(g, ch)
            for ch in cluster_heads
        }

        for node in g.nodes():
            if node in cluster_heads:
                node_to_ch[node] = node
                continue

            best_ch = None
            best_dist = float("inf")
            best_deg = -1

            for ch in cluster_heads:
                dist = sp_maps[ch].get(node, float("inf"))
                deg = g.degree[ch]
                if dist < best_dist or (dist == best_dist and deg > best_deg):
                    best_dist = dist
                    best_deg = deg
                    best_ch = ch

            node_to_ch[node] = best_ch if best_ch is not None else cluster_heads[0]

        return cluster_heads, node_to_ch

    def graph_stats(self, g: nx.Graph) -> Tuple[float, float, float]:
        avg_degree = sum(dict(g.degree()).values()) / g.number_of_nodes()

        if nx.is_connected(g):
            avg_path_length = nx.average_shortest_path_length(g)
        else:
            largest_cc = max(nx.connected_components(g), key=len)
            subg = g.subgraph(largest_cc).copy()
            avg_path_length = nx.average_shortest_path_length(subg)

        clustering_coeff = nx.average_clustering(g)
        return avg_path_length, avg_degree, clustering_coeff

    def reachable_nodes(self, g: nx.Graph, source: int, active_nodes: Set[int]) -> Set[int]:
        if source not in active_nodes:
            return set()

        visited = {source}
        q = deque([source])

        while q:
            u = q.popleft()
            for v in g.neighbors(u):
                if v in active_nodes and v not in visited:
                    visited.add(v)
                    q.append(v)

        return visited

    def make_result(
        self,
        protocol: str,
        g: nx.Graph,
        informed: Set[int],
        source: int,
        active: Set[int],
        rounds: int,
        transmissions: int,
        duplicates: int,
    ) -> SimulationResult:
        reachable = self.reachable_nodes(g, source, active)
        informed_count = len(informed.intersection(reachable))
        reachable_count = len(reachable)

        delivery_ratio = informed_count / reachable_count if reachable_count else 0.0
        duplicate_ratio = duplicates / transmissions if transmissions else 0.0
        propagation_efficiency = informed_count / transmissions if transmissions else 0.0

        avg_path_length, avg_degree, clustering_coeff = self.graph_stats(g)

        return SimulationResult(
            protocol=protocol,
            node_count=g.number_of_nodes(),
            informed_count=informed_count,
            reachable_count=reachable_count,
            delivery_ratio=delivery_ratio,
            rounds=rounds,
            transmissions=transmissions,
            duplicates=duplicates,
            duplicate_ratio=duplicate_ratio,
            propagation_efficiency=propagation_efficiency,
            avg_path_length=avg_path_length,
            avg_degree=avg_degree,
            clustering_coeff=clustering_coeff,
        )

    def run_gossip(
        self,
        g: nx.Graph,
        source: int = 0,
        fanout: int = 2,
        active_nodes: Optional[Set[int]] = None,
        node_capacity: Optional[Dict[int, int]] = None,
        max_rounds: int = 100,
        join_schedule: Optional[Dict[int, List[int]]] = None,
    ) -> SimulationResult:
        active = set(g.nodes()) if active_nodes is None else set(active_nodes)
        join_schedule = join_schedule or {}

        if source not in active:
            return self.make_result("gossip", g, set(), source, active, 0, 0, 0)

        informed: Set[int] = {source}
        frontier: Set[int] = {source}
        transmissions = 0
        duplicates = 0
        rounds = 0

        for round_idx in range(1, max_rounds + 1):
            if round_idx in join_schedule:
                for node in join_schedule[round_idx]:
                    active.add(node)

            next_frontier: Set[int] = set()

            for u in frontier:
                if u not in active:
                    continue

                neighbors = [v for v in g.neighbors(u) if v in active]
                if not neighbors:
                    continue

                k = min(fanout, len(neighbors))
                chosen = self.rng.sample(neighbors, k)

                if node_capacity is not None:
                    cap = node_capacity.get(u, len(chosen))
                    chosen = chosen[:cap]

                for v in chosen:
                    transmissions += 1
                    if v in informed:
                        duplicates += 1
                    else:
                        informed.add(v)
                        next_frontier.add(v)

            rounds = round_idx
            frontier = next_frontier

            reachable = self.reachable_nodes(g, source, active)
            if reachable and reachable.issubset(informed):
                break
            if not frontier and not join_schedule.get(round_idx + 1):
                break

        return self.make_result("gossip", g, informed, source, active, rounds, transmissions, duplicates)

    def run_cluster(
        self,
        g: nx.Graph,
        cluster_heads: List[int],
        node_to_ch: Dict[int, int],
        source: int = 0,
        active_nodes: Optional[Set[int]] = None,
        failed_nodes: Optional[Set[int]] = None,
        overloaded_ch_limit: Optional[int] = None,
        max_rounds: int = 100,
    ) -> SimulationResult:
        active = set(g.nodes()) if active_nodes is None else set(active_nodes)
        failed = set() if failed_nodes is None else set(failed_nodes)
        active -= failed

        if source not in active:
            return self.make_result("cluster", g, set(), source, active, 0, 0, 0)

        transmissions = 0
        duplicates = 0
        informed: Set[int] = {source}
        rounds = 0

        source_ch = node_to_ch[source]
        informed_chs: Set[int] = set()

        if source != source_ch and source_ch in active:
            transmissions += 1
            informed.add(source_ch)
            informed_chs.add(source_ch)
        elif source == source_ch:
            informed_chs.add(source)

        rounds = 1

        active_chs = [ch for ch in cluster_heads if ch in active]
        ch_overlay = nx.Graph()
        ch_overlay.add_nodes_from(active_chs)

        active_subgraph = g.subgraph(active)

        for i in range(len(active_chs)):
            for j in range(i + 1, len(active_chs)):
                u = active_chs[i]
                v = active_chs[j]
                try:
                    dist = nx.shortest_path_length(active_subgraph, u, v)
                    if dist <= 3:
                        ch_overlay.add_edge(u, v)
                except nx.NetworkXNoPath:
                    pass

        if ch_overlay.number_of_edges() == 0 and len(active_chs) > 1:
            for i in range(len(active_chs) - 1):
                ch_overlay.add_edge(active_chs[i], active_chs[i + 1])

        frontier = set(informed_chs)
        visited_chs = set(informed_chs)

        while frontier and len(visited_chs) < len(active_chs) and rounds < max_rounds:
            next_frontier = set()
            for ch in frontier:
                for nbr in ch_overlay.neighbors(ch):
                    if nbr not in active:
                        continue
                    transmissions += 1
                    if nbr in visited_chs:
                        duplicates += 1
                    else:
                        visited_chs.add(nbr)
                        informed.add(nbr)
                        next_frontier.add(nbr)
            frontier = next_frontier
            rounds += 1

        informed_chs = visited_chs

        members_by_ch: Dict[int, List[int]] = defaultdict(list)
        for node, ch in node_to_ch.items():
            if node != ch:
                members_by_ch[ch].append(node)

        rounds += 1
        for ch in informed_chs:
            if ch not in active:
                continue

            members = [m for m in members_by_ch[ch] if m in active]
            if overloaded_ch_limit is not None:
                members = members[:overloaded_ch_limit]

            for m in members:
                transmissions += 1
                if m in informed:
                    duplicates += 1
                else:
                    informed.add(m)

        return self.make_result("cluster", g, informed, source, active, rounds, transmissions, duplicates)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def build_results_dirname() -> str:
    timestamp = datetime.now().strftime("%Y%b%d_%H%M")
    return f"results_{timestamp}"


def ensure_new_dir(path: str) -> None:
    if os.path.exists(path):
        print(f"ERROR: results folder already exists: {os.path.abspath(path)}")
        print("Stopping experiment to avoid overwriting existing analysis.")
        sys.exit(1)
    os.makedirs(path, exist_ok=False)


def write_csv(path: str, rows: List[dict]) -> None:
    if not rows:
        return
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def mean_of(results: List[SimulationResult], attr: str) -> float:
    vals = [getattr(r, attr) for r in results]
    return sum(vals) / len(vals) if vals else 0.0


def summarize_results(experiment_id: int, extra: Dict, results: List[SimulationResult]) -> dict:
    return {
        "experiment": experiment_id,
        **extra,
        "avg_rounds": round(mean_of(results, "rounds"), 4),
        "avg_transmissions": round(mean_of(results, "transmissions"), 4),
        "avg_duplicates": round(mean_of(results, "duplicates"), 4),
        "avg_duplicate_ratio": round(mean_of(results, "duplicate_ratio"), 4),
        "avg_delivery_ratio": round(mean_of(results, "delivery_ratio"), 4),
        "avg_propagation_efficiency": round(mean_of(results, "propagation_efficiency"), 4),
        "avg_path_length": round(mean_of(results, "avg_path_length"), 4),
        "avg_degree": round(mean_of(results, "avg_degree"), 4),
        "avg_clustering_coeff": round(mean_of(results, "clustering_coeff"), 4),
    }


# -----------------------------------------------------------------------------
# Experiments
# -----------------------------------------------------------------------------

def experiment_1_fanout_vs_duplication(out_dir: str, runs: int = RUNS) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    ba_m = 3

    for fanout in [1, 2, 3, 4, 5, 6, 8]:
        run_results = []

        for i in range(runs):
            seed = BASE_SEED + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            res = sim.run_gossip(g=g, source=0, fanout=fanout)
            run_results.append(res)

        rows.append(
            summarize_results(1, {"fanout": fanout, "node_count": node_count, "ba_m": ba_m}, run_results)
        )

    write_csv(os.path.join(out_dir, "exp1_fanout_vs_duplication.csv"), rows)
    return rows


def experiment_2_ch_count_vs_node_count(out_dir: str, runs: int = RUNS) -> List[dict]:
    rows: List[dict] = []

    for node_count in [50, 100, 200, 300, 400]:
        ba_m = 3
        ch_options = sorted(set([2, 4, 6, 8, max(2, node_count // 20), max(2, node_count // 10)]))
        ch_options = [c for c in ch_options if c < node_count]

        for ch_count in ch_options:
            run_results = []

            for i in range(runs):
                seed = BASE_SEED + i
                sim = BASimulator(seed)
                g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
                chs, node_to_ch = sim.derive_clusters_from_graph(g, ch_count=ch_count)
                res = sim.run_cluster(g=g, cluster_heads=chs, node_to_ch=node_to_ch, source=0)
                run_results.append(res)

            rows.append(
                summarize_results(2, {"node_count": node_count, "ch_count": ch_count, "ba_m": ba_m}, run_results)
            )

    write_csv(os.path.join(out_dir, "exp2_ch_count_vs_node_count.csv"), rows)
    return rows


def experiment_3_topology_density_vs_performance(out_dir: str, runs: int = RUNS) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    fanout = 3

    for ba_m in [1, 2, 3, 4, 6, 8]:
        run_results = []

        for i in range(runs):
            seed = BASE_SEED + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            res = sim.run_gossip(g=g, source=0, fanout=fanout)
            run_results.append(res)

        rows.append(
            summarize_results(3, {"node_count": node_count, "ba_m": ba_m, "fanout": fanout}, run_results)
        )

    write_csv(os.path.join(out_dir, "exp3_topology_density_vs_performance.csv"), rows)
    return rows


def experiment_4_ch_overload_failure(out_dir: str, runs: int = RUNS) -> List[dict]:
    rows: List[dict] = []
    node_count = 120
    ch_count = 6
    ba_m = 3

    for overload_limit in [None, 20, 10, 5, 2]:
        run_results = []

        for i in range(runs):
            seed = BASE_SEED + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            chs, node_to_ch = sim.derive_clusters_from_graph(g, ch_count=ch_count)
            res = sim.run_cluster(
                g=g,
                cluster_heads=chs,
                node_to_ch=node_to_ch,
                source=0,
                overloaded_ch_limit=overload_limit,
            )
            run_results.append(res)

        rows.append(
            summarize_results(
                4,
                {
                    "scenario": "overload",
                    "node_count": node_count,
                    "ch_count": ch_count,
                    "ba_m": ba_m,
                    "overload_limit": "full" if overload_limit is None else overload_limit,
                    "failed_chs": 0,
                },
                run_results,
            )
        )

    for failed_ch_count in [0, 1, 2, 3]:
        run_results = []

        for i in range(runs):
            seed = BASE_SEED + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            chs, node_to_ch = sim.derive_clusters_from_graph(g, ch_count=ch_count)

            failed_nodes = set(chs[:failed_ch_count])
            if 0 in failed_nodes:
                failed_nodes.remove(0)

            res = sim.run_cluster(
                g=g,
                cluster_heads=chs,
                node_to_ch=node_to_ch,
                source=0,
                failed_nodes=failed_nodes,
            )
            run_results.append(res)

        rows.append(
            summarize_results(
                4,
                {
                    "scenario": "failure",
                    "node_count": node_count,
                    "ch_count": ch_count,
                    "ba_m": ba_m,
                    "overload_limit": "full",
                    "failed_chs": failed_ch_count,
                },
                run_results,
            )
        )

    write_csv(os.path.join(out_dir, "exp4_ch_overload_failure.csv"), rows)
    return rows


def experiment_5_churn_sensitivity(out_dir: str, runs: int = RUNS) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    ba_m = 3
    ch_count = 5
    fanout = 3

    for churn_rate in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
        gossip_runs = []
        cluster_runs = []

        for i in range(runs):
            seed = BASE_SEED + i
            sim = BASimulator(seed)

            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            chs, node_to_ch = sim.derive_clusters_from_graph(g, ch_count=ch_count)

            all_nodes = list(g.nodes())
            remove_count = int(churn_rate * node_count)
            removed = set(sim.rng.sample(all_nodes[1:], min(remove_count, node_count - 1)))
            active = set(all_nodes) - removed

            gossip_runs.append(
                sim.run_gossip(g=g, source=0, fanout=fanout, active_nodes=active)
            )

            sim_cluster = BASimulator(seed)
            cluster_runs.append(
                sim_cluster.run_cluster(
                    g=g,
                    cluster_heads=chs,
                    node_to_ch=node_to_ch,
                    source=0,
                    active_nodes=active,
                )
            )

        rows.append(
            summarize_results(
                5,
                {"protocol": "gossip", "node_count": node_count, "churn_rate": churn_rate, "ba_m": ba_m},
                gossip_runs,
            )
        )

        rows.append(
            summarize_results(
                5,
                {"protocol": "cluster", "node_count": node_count, "churn_rate": churn_rate, "ba_m": ba_m},
                cluster_runs,
            )
        )

    write_csv(os.path.join(out_dir, "exp5_churn_sensitivity.csv"), rows)
    return rows


def experiment_6_heterogeneous_resources(out_dir: str, runs: int = RUNS) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    ba_m = 3
    fanout = 4

    for scenario in ["homogeneous", "heterogeneous"]:
        run_results = []

        for i in range(runs):
            seed = BASE_SEED + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)

            if scenario == "homogeneous":
                node_capacity = {node: 3 for node in g.nodes()}
            else:
                node_capacity = {}
                for node in g.nodes():
                    r = sim.rng.random()
                    if r < 0.2:
                        node_capacity[node] = 4
                    elif r < 0.7:
                        node_capacity[node] = 2
                    else:
                        node_capacity[node] = 1

            res = sim.run_gossip(g=g, source=0, fanout=fanout, node_capacity=node_capacity)
            run_results.append(res)

        rows.append(
            summarize_results(
                6,
                {"scenario": scenario, "node_count": node_count, "fanout": fanout, "ba_m": ba_m},
                run_results,
            )
        )

    write_csv(os.path.join(out_dir, "exp6_heterogeneous_resources.csv"), rows)
    return rows


# -----------------------------------------------------------------------------
# Printing
# -----------------------------------------------------------------------------

def print_summary(title: str, rows: List[dict], limit: int = 8) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    for row in rows[:limit]:
        print(row)
    if len(rows) > limit:
        print(f"... ({len(rows) - limit} more rows)")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> None:
    out_dir = build_results_dirname()
    ensure_new_dir(out_dir)

    exp1 = experiment_1_fanout_vs_duplication(out_dir=out_dir, runs=RUNS)
    exp2 = experiment_2_ch_count_vs_node_count(out_dir=out_dir, runs=RUNS)
    exp3 = experiment_3_topology_density_vs_performance(out_dir=out_dir, runs=RUNS)
    exp4 = experiment_4_ch_overload_failure(out_dir=out_dir, runs=RUNS)
    exp5 = experiment_5_churn_sensitivity(out_dir=out_dir, runs=RUNS)
    exp6 = experiment_6_heterogeneous_resources(out_dir=out_dir, runs=RUNS)

    print_summary("Exp 1 - Fanout vs Duplication", exp1)
    print_summary("Exp 2 - CH Count vs Node Count", exp2)
    print_summary("Exp 3 - Topology Density vs Performance", exp3)
    print_summary("Exp 4 - CH Overload / Failure", exp4)
    print_summary("Exp 5 - Churn Sensitivity", exp5)
    print_summary("Exp 6 - Heterogeneous Resources", exp6)

    print("\nCSV files written to:", os.path.abspath(out_dir))
    print(f"RUNS = {RUNS}, BASE_SEED = {BASE_SEED}")


if __name__ == "__main__":
    main()