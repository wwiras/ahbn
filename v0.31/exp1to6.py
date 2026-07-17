from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Set, Tuple, Optional


# -----------------------------------------------------------------------------
# Data class
# -----------------------------------------------------------------------------

@dataclass
class SimulationResult:
    protocol: str
    node_count: int
    informed_count: int
    delivery_ratio: float
    rounds: int
    transmissions: int
    duplicates: int
    duplicate_ratio: float
    reachable_count: int


# -----------------------------------------------------------------------------
# Core simulator
# -----------------------------------------------------------------------------

class PlainBroadcastSimulator:
    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = random.Random(seed)

    # -------------------------
    # Topology builders
    # -------------------------

    def build_random_graph(self, node_count: int, degree_hint: int) -> Dict[int, Set[int]]:
        """
        Build an undirected random graph with approximate average degree.
        First ensures connectivity using a chain, then adds random edges.
        """
        if node_count < 2:
            return {0: set()}

        graph: Dict[int, Set[int]] = {i: set() for i in range(node_count)}

        # ensure connectivity
        for i in range(node_count - 1):
            graph[i].add(i + 1)
            graph[i + 1].add(i)

        target_edges = max(node_count - 1, int(node_count * degree_hint / 2))
        current_edges = node_count - 1
        attempts = 0
        max_attempts = target_edges * 20 + 100

        while current_edges < target_edges and attempts < max_attempts:
            u = self.rng.randrange(node_count)
            v = self.rng.randrange(node_count)
            attempts += 1
            if u == v or v in graph[u]:
                continue
            graph[u].add(v)
            graph[v].add(u)
            current_edges += 1

        return graph

    def build_clusters(
        self,
        node_count: int,
        ch_count: int,
        intra_degree: int = 2,
        inter_ch_degree: int = 2,
    ) -> Tuple[Dict[int, Set[int]], List[int], Dict[int, int]]:
        """
        Build a clustered graph.
        Returns:
            graph, cluster_heads, node_to_ch
        """
        ch_count = max(1, min(ch_count, node_count))
        nodes = list(range(node_count))
        cluster_heads = list(range(ch_count))
        graph: Dict[int, Set[int]] = {i: set() for i in nodes}
        node_to_ch: Dict[int, int] = {}

        non_heads = [n for n in nodes if n not in cluster_heads]

        # assign members evenly to CHs
        for idx, node in enumerate(non_heads):
            ch = cluster_heads[idx % ch_count]
            node_to_ch[node] = ch
            node_to_ch[ch] = ch
            graph[node].add(ch)
            graph[ch].add(node)

        for ch in cluster_heads:
            node_to_ch[ch] = ch

        groups: Dict[int, List[int]] = defaultdict(list)
        for node, ch in node_to_ch.items():
            groups[ch].append(node)

        # add intra-cluster member-member edges
        for ch, members in groups.items():
            members_only = [m for m in members if m != ch]
            for u in members_only:
                candidates = [v for v in members_only if v != u and v not in graph[u]]
                self.rng.shuffle(candidates)
                for v in candidates[:intra_degree]:
                    graph[u].add(v)
                    graph[v].add(u)

        # connect CHs in a ring
        if len(cluster_heads) > 1:
            for i in range(len(cluster_heads)):
                u = cluster_heads[i]
                v = cluster_heads[(i + 1) % len(cluster_heads)]
                graph[u].add(v)
                graph[v].add(u)

        # add extra CH-CH edges
        extra_target = max(len(cluster_heads), int(len(cluster_heads) * inter_ch_degree / 2))
        current_ch_edges = self._count_subgraph_edges(graph, set(cluster_heads))
        attempts = 0
        while current_ch_edges < extra_target and attempts < extra_target * 20 + 50:
            if len(cluster_heads) < 2:
                break
            u, v = self.rng.sample(cluster_heads, 2)
            attempts += 1
            if u == v or v in graph[u]:
                continue
            graph[u].add(v)
            graph[v].add(u)
            current_ch_edges += 1

        return graph, cluster_heads, node_to_ch

    def _count_subgraph_edges(self, graph: Dict[int, Set[int]], subset: Set[int]) -> int:
        count = 0
        for u in subset:
            for v in graph[u]:
                if v in subset and u < v:
                    count += 1
        return count

    # -------------------------
    # Graph helper
    # -------------------------

    def reachable_nodes(self, graph: Dict[int, Set[int]], source: int, active_nodes: Set[int]) -> Set[int]:
        if source not in active_nodes:
            return set()
        stack = [source]
        seen = {source}
        while stack:
            u = stack.pop()
            for v in graph[u]:
                if v in active_nodes and v not in seen:
                    seen.add(v)
                    stack.append(v)
        return seen

    # -------------------------
    # Gossip simulation
    # -------------------------

    def run_gossip(
        self,
        graph: Dict[int, Set[int]],
        source: int = 0,
        fanout: int = 2,
        active_nodes: Optional[Set[int]] = None,
        node_capacity: Optional[Dict[int, int]] = None,
        max_rounds: int = 100,
        join_schedule: Optional[Dict[int, List[int]]] = None,
    ) -> SimulationResult:
        node_count = len(graph)
        active = set(range(node_count)) if active_nodes is None else set(active_nodes)
        join_schedule = join_schedule or {}

        if source not in active:
            return SimulationResult("gossip", node_count, 0, 0.0, 0, 0, 0, 0.0, 0)

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

                neighbors = [v for v in graph[u] if v in active]
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

            reachable = self.reachable_nodes(graph, source, active)
            if reachable and reachable.issubset(informed):
                break
            if not frontier and not join_schedule.get(round_idx + 1):
                break

        reachable = self.reachable_nodes(graph, source, active)
        informed_count = len(informed.intersection(reachable))
        reachable_count = len(reachable)
        delivery_ratio = informed_count / reachable_count if reachable_count else 0.0
        duplicate_ratio = duplicates / transmissions if transmissions else 0.0

        return SimulationResult(
            protocol="gossip",
            node_count=node_count,
            informed_count=informed_count,
            delivery_ratio=delivery_ratio,
            rounds=rounds,
            transmissions=transmissions,
            duplicates=duplicates,
            duplicate_ratio=duplicate_ratio,
            reachable_count=reachable_count,
        )

    # -------------------------
    # Cluster simulation
    # -------------------------

    def run_cluster(
        self,
        graph: Dict[int, Set[int]],
        cluster_heads: List[int],
        node_to_ch: Dict[int, int],
        source: int = 0,
        active_nodes: Optional[Set[int]] = None,
        failed_nodes: Optional[Set[int]] = None,
        overloaded_ch_limit: Optional[int] = None,
        max_rounds: int = 100,
    ) -> SimulationResult:
        """
        Structured dissemination:
        1. source -> own CH
        2. CH overlay dissemination
        3. CH -> cluster members

        Supports:
        - failed_nodes
        - overloaded_ch_limit
        """
        node_count = len(graph)
        active = set(range(node_count)) if active_nodes is None else set(active_nodes)
        failed = set() if failed_nodes is None else set(failed_nodes)
        active -= failed

        if source not in active:
            return SimulationResult("cluster", node_count, 0, 0.0, 0, 0, 0, 0.0, 0)

        transmissions = 0
        duplicates = 0
        informed: Set[int] = {source}
        rounds = 0

        source_ch = node_to_ch[source]
        informed_chs: Set[int] = set()

        # phase 1: source -> CH
        if source != source_ch and source_ch in active:
            transmissions += 1
            informed.add(source_ch)
            informed_chs.add(source_ch)
        elif source == source_ch:
            informed_chs.add(source)

        rounds = 1

        # phase 2: CH overlay dissemination
        active_chs = [ch for ch in cluster_heads if ch in active]
        frontier = set(informed_chs)
        visited_chs = set(informed_chs)

        while frontier and len(visited_chs) < len(active_chs) and rounds < max_rounds:
            next_frontier = set()
            for ch in frontier:
                for nbr in graph[ch]:
                    if nbr not in active or nbr not in cluster_heads:
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

        # phase 3: CH -> members
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

        reachable = self.reachable_nodes(graph, source, active)
        informed_count = len(informed.intersection(reachable))
        reachable_count = len(reachable)
        delivery_ratio = informed_count / reachable_count if reachable_count else 0.0
        duplicate_ratio = duplicates / transmissions if transmissions else 0.0

        return SimulationResult(
            protocol="cluster",
            node_count=node_count,
            informed_count=informed_count,
            delivery_ratio=delivery_ratio,
            rounds=rounds,
            transmissions=transmissions,
            duplicates=duplicates,
            duplicate_ratio=duplicate_ratio,
            reachable_count=reachable_count,
        )


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


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


# -----------------------------------------------------------------------------
# Experiment 1
# -----------------------------------------------------------------------------

def experiment_1_fanout_vs_duplication(out_dir: str = "results", runs: int = 20) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    degree_hint = 6

    for fanout in [1, 2, 3, 4, 5, 6, 8]:
        run_results = []
        for seed in range(runs):
            sim = PlainBroadcastSimulator(seed)
            graph = sim.build_random_graph(node_count=node_count, degree_hint=degree_hint)
            res = sim.run_gossip(graph=graph, source=0, fanout=fanout)
            run_results.append(res)

        rows.append({
            "experiment": 1,
            "fanout": fanout,
            "node_count": node_count,
            "avg_rounds": round(mean_of(run_results, "rounds"), 4),
            "avg_transmissions": round(mean_of(run_results, "transmissions"), 4),
            "avg_duplicates": round(mean_of(run_results, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(run_results, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(run_results, "delivery_ratio"), 4),
        })

    write_csv(os.path.join(out_dir, "exp1_fanout_vs_duplication.csv"), rows)
    return rows


# -----------------------------------------------------------------------------
# Experiment 2
# -----------------------------------------------------------------------------

def experiment_2_ch_count_vs_node_count(out_dir: str = "results", runs: int = 20) -> List[dict]:
    rows: List[dict] = []

    for node_count in [50, 100, 200, 300]:
        ch_options = sorted(set([2, 4, 6, 8, max(2, node_count // 20), max(2, node_count // 10)]))
        ch_options = [c for c in ch_options if c < node_count]

        for ch_count in ch_options:
            run_results = []
            for seed in range(runs):
                sim = PlainBroadcastSimulator(seed)
                graph, chs, node_to_ch = sim.build_clusters(node_count=node_count, ch_count=ch_count)
                res = sim.run_cluster(graph=graph, cluster_heads=chs, node_to_ch=node_to_ch, source=0)
                run_results.append(res)

            rows.append({
                "experiment": 2,
                "node_count": node_count,
                "ch_count": ch_count,
                "avg_rounds": round(mean_of(run_results, "rounds"), 4),
                "avg_transmissions": round(mean_of(run_results, "transmissions"), 4),
                "avg_duplicates": round(mean_of(run_results, "duplicates"), 4),
                "avg_duplicate_ratio": round(mean_of(run_results, "duplicate_ratio"), 4),
                "avg_delivery_ratio": round(mean_of(run_results, "delivery_ratio"), 4),
            })

    write_csv(os.path.join(out_dir, "exp2_ch_count_vs_node_count.csv"), rows)
    return rows


# -----------------------------------------------------------------------------
# Experiment 3
# -----------------------------------------------------------------------------

def experiment_3_topology_density_vs_performance(out_dir: str = "results", runs: int = 20) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    fanout = 3

    for degree_hint in [2, 4, 6, 8, 12, 16]:
        run_results = []
        for seed in range(runs):
            sim = PlainBroadcastSimulator(seed)
            graph = sim.build_random_graph(node_count=node_count, degree_hint=degree_hint)
            res = sim.run_gossip(graph=graph, source=0, fanout=fanout)
            run_results.append(res)

        rows.append({
            "experiment": 3,
            "node_count": node_count,
            "degree_hint": degree_hint,
            "fanout": fanout,
            "avg_rounds": round(mean_of(run_results, "rounds"), 4),
            "avg_transmissions": round(mean_of(run_results, "transmissions"), 4),
            "avg_duplicates": round(mean_of(run_results, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(run_results, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(run_results, "delivery_ratio"), 4),
        })

    write_csv(os.path.join(out_dir, "exp3_topology_density_vs_performance.csv"), rows)
    return rows


# -----------------------------------------------------------------------------
# Experiment 4
# -----------------------------------------------------------------------------

def experiment_4_ch_overload_failure(out_dir: str = "results", runs: int = 20) -> List[dict]:
    rows: List[dict] = []
    node_count = 120
    ch_count = 6

    # overload scenario
    for overload_limit in [None, 20, 10, 5, 2]:
        run_results = []
        for seed in range(runs):
            sim = PlainBroadcastSimulator(seed)
            graph, chs, node_to_ch = sim.build_clusters(node_count=node_count, ch_count=ch_count)
            res = sim.run_cluster(
                graph=graph,
                cluster_heads=chs,
                node_to_ch=node_to_ch,
                source=0,
                overloaded_ch_limit=overload_limit,
            )
            run_results.append(res)

        rows.append({
            "experiment": 4,
            "scenario": "overload",
            "node_count": node_count,
            "ch_count": ch_count,
            "overload_limit": "full" if overload_limit is None else overload_limit,
            "failed_chs": 0,
            "avg_rounds": round(mean_of(run_results, "rounds"), 4),
            "avg_transmissions": round(mean_of(run_results, "transmissions"), 4),
            "avg_duplicates": round(mean_of(run_results, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(run_results, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(run_results, "delivery_ratio"), 4),
        })

    # failure scenario
    for failed_ch_count in [0, 1, 2, 3]:
        run_results = []
        for seed in range(runs):
            sim = PlainBroadcastSimulator(seed)
            graph, chs, node_to_ch = sim.build_clusters(node_count=node_count, ch_count=ch_count)
            failed_nodes = set(chs[:failed_ch_count])

            # keep source CH alive if source=0 happens to be a CH
            if 0 in failed_nodes:
                failed_nodes.remove(0)

            res = sim.run_cluster(
                graph=graph,
                cluster_heads=chs,
                node_to_ch=node_to_ch,
                source=0,
                failed_nodes=failed_nodes,
            )
            run_results.append(res)

        rows.append({
            "experiment": 4,
            "scenario": "failure",
            "node_count": node_count,
            "ch_count": ch_count,
            "overload_limit": "full",
            "failed_chs": failed_ch_count,
            "avg_rounds": round(mean_of(run_results, "rounds"), 4),
            "avg_transmissions": round(mean_of(run_results, "transmissions"), 4),
            "avg_duplicates": round(mean_of(run_results, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(run_results, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(run_results, "delivery_ratio"), 4),
        })

    write_csv(os.path.join(out_dir, "exp4_ch_overload_failure.csv"), rows)
    return rows


# -----------------------------------------------------------------------------
# Experiment 5
# -----------------------------------------------------------------------------

def experiment_5_churn_sensitivity(out_dir: str = "results", runs: int = 20) -> List[dict]:
    rows: List[dict] = []
    node_count = 100

    for churn_rate in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
        gossip_runs = []
        cluster_runs = []

        for seed in range(runs):
            sim = PlainBroadcastSimulator(seed)
            graph = sim.build_random_graph(node_count=node_count, degree_hint=6)
            all_nodes = list(range(node_count))
            remove_count = int(churn_rate * node_count)
            removed = set(sim.rng.sample(all_nodes[1:], min(remove_count, node_count - 1)))
            active = set(all_nodes) - removed
            gossip_runs.append(sim.run_gossip(graph=graph, source=0, fanout=3, active_nodes=active))

            sim2 = PlainBroadcastSimulator(seed)
            cgraph, chs, node_to_ch = sim2.build_clusters(node_count=node_count, ch_count=5)
            all_nodes2 = list(range(node_count))
            removed2 = set(sim2.rng.sample(all_nodes2[1:], min(remove_count, node_count - 1)))
            active2 = set(all_nodes2) - removed2
            cluster_runs.append(
                sim2.run_cluster(
                    graph=cgraph,
                    cluster_heads=chs,
                    node_to_ch=node_to_ch,
                    source=0,
                    active_nodes=active2,
                )
            )

        rows.append({
            "experiment": 5,
            "protocol": "gossip",
            "node_count": node_count,
            "churn_rate": churn_rate,
            "avg_rounds": round(mean_of(gossip_runs, "rounds"), 4),
            "avg_transmissions": round(mean_of(gossip_runs, "transmissions"), 4),
            "avg_duplicates": round(mean_of(gossip_runs, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(gossip_runs, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(gossip_runs, "delivery_ratio"), 4),
        })

        rows.append({
            "experiment": 5,
            "protocol": "cluster",
            "node_count": node_count,
            "churn_rate": churn_rate,
            "avg_rounds": round(mean_of(cluster_runs, "rounds"), 4),
            "avg_transmissions": round(mean_of(cluster_runs, "transmissions"), 4),
            "avg_duplicates": round(mean_of(cluster_runs, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(cluster_runs, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(cluster_runs, "delivery_ratio"), 4),
        })

    write_csv(os.path.join(out_dir, "exp5_churn_sensitivity.csv"), rows)
    return rows


# -----------------------------------------------------------------------------
# Experiment 6
# -----------------------------------------------------------------------------

def experiment_6_heterogeneous_resources(out_dir: str = "results", runs: int = 20) -> List[dict]:
    rows: List[dict] = []
    node_count = 100
    degree_hint = 6
    fanout = 4

    for scenario in ["homogeneous", "heterogeneous"]:
        run_results = []
        for seed in range(runs):
            sim = PlainBroadcastSimulator(seed)
            graph = sim.build_random_graph(node_count=node_count, degree_hint=degree_hint)

            if scenario == "homogeneous":
                node_capacity = {i: 3 for i in range(node_count)}
            else:
                node_capacity = {}
                for i in range(node_count):
                    r = sim.rng.random()
                    if r < 0.2:
                        node_capacity[i] = 4
                    elif r < 0.7:
                        node_capacity[i] = 2
                    else:
                        node_capacity[i] = 1

            res = sim.run_gossip(
                graph=graph,
                source=0,
                fanout=fanout,
                node_capacity=node_capacity,
            )
            run_results.append(res)

        rows.append({
            "experiment": 6,
            "scenario": scenario,
            "node_count": node_count,
            "fanout": fanout,
            "avg_rounds": round(mean_of(run_results, "rounds"), 4),
            "avg_transmissions": round(mean_of(run_results, "transmissions"), 4),
            "avg_duplicates": round(mean_of(run_results, "duplicates"), 4),
            "avg_duplicate_ratio": round(mean_of(run_results, "duplicate_ratio"), 4),
            "avg_delivery_ratio": round(mean_of(run_results, "delivery_ratio"), 4),
        })

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
    out_dir = "results"
    ensure_dir(out_dir)

    exp1 = experiment_1_fanout_vs_duplication(out_dir=out_dir, runs=20)
    exp2 = experiment_2_ch_count_vs_node_count(out_dir=out_dir, runs=20)
    exp3 = experiment_3_topology_density_vs_performance(out_dir=out_dir, runs=20)
    exp4 = experiment_4_ch_overload_failure(out_dir=out_dir, runs=20)
    exp5 = experiment_5_churn_sensitivity(out_dir=out_dir, runs=20)
    exp6 = experiment_6_heterogeneous_resources(out_dir=out_dir, runs=20)

    print_summary("Exp 1 - Fanout vs Duplication", exp1)
    print_summary("Exp 2 - CH Count vs Node Count", exp2)
    print_summary("Exp 3 - Topology Density vs Performance", exp3)
    print_summary("Exp 4 - CH Overload / Failure", exp4)
    print_summary("Exp 5 - Churn Sensitivity", exp5)
    print_summary("Exp 6 - Heterogeneous Resources", exp6)

    print("\nCSV files written to:", os.path.abspath(out_dir))


if __name__ == "__main__":
    main()