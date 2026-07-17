from __future__ import annotations

import csv
import os
import sys
import math
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
# Logging helpers
# -----------------------------------------------------------------------------

def log(msg: str) -> None:
    print(f"[INFO] {msg}", flush=True)


def log_exp_start(exp_id: int, title: str) -> None:
    print(f"\n[INFO] Experiment {exp_id} ongoing: {title}", flush=True)


def log_exp_done(exp_id: int, title: str, out_file: str) -> None:
    print(f"[INFO] Experiment {exp_id} done: {title}", flush=True)
    print(f"[INFO] Output written: {out_file}", flush=True)


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

    # -------------------------------------------------------------------------
    # Enhanced Exp 2 helpers
    # -------------------------------------------------------------------------

    def assign_clusters_fixed(
        self,
        node_count: int,
        ch_count: int,
    ) -> Tuple[List[int], Dict[int, List[int]], Dict[int, int]]:
        all_nodes = list(range(node_count))
        ch_nodes = list(range(ch_count))

        members_by_ch: Dict[int, List[int]] = {ch: [] for ch in ch_nodes}
        node_to_ch: Dict[int, int] = {}

        member_nodes = [n for n in all_nodes if n not in ch_nodes]
        for idx, node in enumerate(member_nodes):
            ch = ch_nodes[idx % ch_count]
            members_by_ch[ch].append(node)
            node_to_ch[node] = ch

        for ch in ch_nodes:
            node_to_ch[ch] = ch

        return ch_nodes, members_by_ch, node_to_ch

    def build_sparse_ch_overlay_fixed(
        self,
        ch_nodes: List[int],
        target_degree: int = 2,
    ) -> Dict[int, Set[int]]:
        overlay: Dict[int, Set[int]] = {ch: set() for ch in ch_nodes}
        c = len(ch_nodes)

        if c <= 1:
            return overlay

        for i in range(c):
            a = ch_nodes[i]
            b = ch_nodes[(i + 1) % c]
            overlay[a].add(b)
            overlay[b].add(a)

        for ch in ch_nodes:
            while len(overlay[ch]) < min(target_degree, c - 1):
                other = self.rng.choice(ch_nodes)
                if other != ch:
                    overlay[ch].add(other)
                    overlay[other].add(ch)

        return overlay

    def build_intra_cluster_graph_fixed(
        self,
        ch: int,
        members: List[int],
        member_degree: int = 2,
    ) -> Dict[int, Set[int]]:
        nodes = [ch] + members
        g: Dict[int, Set[int]] = {n: set() for n in nodes}

        if not members:
            return g

        seeds = members[:max(1, min(len(members), member_degree))]
        for m in seeds:
            g[ch].add(m)
            g[m].add(ch)

        if len(members) > 1:
            for i in range(len(members)):
                a = members[i]
                b = members[(i + 1) % len(members)]
                g[a].add(b)
                g[b].add(a)

        for m in members:
            while len([x for x in g[m] if x != ch]) < min(member_degree, max(0, len(members) - 1)):
                other = self.rng.choice(members)
                if other != m:
                    g[m].add(other)
                    g[other].add(m)

        return g

    def run_cluster_fixed_exp2(
        self,
        g: nx.Graph,
        source: int = 0,
        ch_count: int = 4,
        ch_service_limit: Optional[int] = None,
        ch_overlay_degree: int = 2,
        member_degree: int = 2,
        member_forward_limit: int = 2,
        transmission_success_prob: float = 1.0,
        max_rounds: int = 100,
    ) -> Tuple[SimulationResult, float]:
        node_count = g.number_of_nodes()

        ch_nodes, members_by_ch, node_to_ch = self.assign_clusters_fixed(node_count=node_count, ch_count=ch_count)
        ch_overlay = self.build_sparse_ch_overlay_fixed(ch_nodes, target_degree=ch_overlay_degree)

        cluster_graphs = {
            ch: self.build_intra_cluster_graph_fixed(ch, members_by_ch[ch], member_degree=member_degree)
            for ch in ch_nodes
        }

        if ch_service_limit is None:
            avg_cluster_size = (node_count - ch_count) / ch_count if ch_count > 0 else 0
            # stricter than before so few-CH cases become more overloaded
            ch_service_limit = max(1, math.ceil(math.log2(max(2, avg_cluster_size + 1))))

        informed: Set[int] = {source}
        active: Set[int] = set(g.nodes())

        seen_edges: Set[Tuple[int, int]] = set()
        rounds = 0
        transmissions = 0
        duplicates = 0

        pending_by_sender: Dict[int, deque] = defaultdict(deque)
        source_ch = node_to_ch[source]

        if source != source_ch:
            pending_by_sender[source].append(source_ch)
        else:
            # usually source should not be CH in Exp 2, but keep safe fallback
            for nb in ch_overlay[source]:
                pending_by_sender[source].append(nb)
            for nb in cluster_graphs[source][source]:
                pending_by_sender[source].append(nb)

        while rounds < max_rounds:
            rounds += 1
            activated_this_round: Set[int] = set()

            active_senders = list(pending_by_sender.keys())
            if not active_senders:
                break

            new_pending: Dict[int, deque] = defaultdict(deque)

            for sender in active_senders:
                sender_is_ch = sender in ch_nodes
                limit = ch_service_limit if sender_is_ch else member_forward_limit
                served = 0

                while pending_by_sender[sender] and served < limit:
                    receiver = pending_by_sender[sender].popleft()
                    served += 1

                    edge = (sender, receiver)
                    if edge in seen_edges:
                        duplicates += 1
                    seen_edges.add(edge)

                    transmissions += 1

                    if self.rng.random() > transmission_success_prob:
                        continue

                    was_new = receiver not in informed
                    if receiver in informed:
                        duplicates += 1
                    else:
                        informed.add(receiver)
                        activated_this_round.add(receiver)

                    if was_new:
                        receiver_ch = node_to_ch[receiver]

                        if receiver in ch_nodes:
                            # prioritize overlay first, then local cluster spread
                            overlay_neighs = list(ch_overlay[receiver])
                            local_neighs = list(cluster_graphs[receiver][receiver])
                            self.rng.shuffle(overlay_neighs)
                            self.rng.shuffle(local_neighs)

                            for nb in overlay_neighs:
                                new_pending[receiver].append(nb)
                            for nb in local_neighs:
                                new_pending[receiver].append(nb)

                        else:
                            local_graph = cluster_graphs[receiver_ch]
                            neighs = list(local_graph[receiver])
                            self.rng.shuffle(neighs)
                            for nb in neighs:
                                new_pending[receiver].append(nb)

                            if receiver_ch not in informed:
                                new_pending[receiver].append(receiver_ch)

                while pending_by_sender[sender]:
                    new_pending[sender].append(pending_by_sender[sender].popleft())

            pending_by_sender = new_pending

            if not activated_this_round and not pending_by_sender:
                break

            if len(informed) == node_count:
                break

        result = self.make_result(
            protocol="cluster_fixed",
            g=g,
            informed=informed,
            source=source,
            active=active,
            rounds=rounds,
            transmissions=transmissions,
            duplicates=duplicates,
        )
        return result, float(ch_service_limit)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------

def build_results_dirname() -> str:
    timestamp = datetime.now().strftime("%Y%b%d_%H%M")
    return f"results_{timestamp}"


def ensure_new_dir(path: str) -> None:
    if os.path.exists(path):
        print(f"[ERROR] Results folder already exists: {os.path.abspath(path)}", flush=True)
        print("[ERROR] Stopping experiment to avoid overwriting existing analysis.", flush=True)
        sys.exit(1)
    os.makedirs(path, exist_ok=False)
    log(f"Results folder created: {os.path.abspath(path)}")


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
        "runs": len(results),
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

def experiment_1_fanout_vs_duplication(out_dir: str, runs: int = RUNS, base_seed: int = BASE_SEED) -> List[dict]:
    title = "Fanout vs Duplication"
    out_file = os.path.join(out_dir, "exp1_fanout_vs_duplication.csv")
    log_exp_start(1, title)

    rows: List[dict] = []
    node_count = 100
    ba_m = 3

    for fanout in [1, 2, 3, 4, 5, 6, 8]:
        run_results = []
        for i in range(runs):
            seed = base_seed + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            res = sim.run_gossip(g=g, source=0, fanout=fanout)
            run_results.append(res)

        rows.append(
            summarize_results(1, {"fanout": fanout, "node_count": node_count, "ba_m": ba_m}, run_results)
        )

    write_csv(out_file, rows)
    log_exp_done(1, title, out_file)
    return rows


def experiment_2_ch_count_vs_node_count(out_dir: str, runs: int = RUNS, base_seed: int = BASE_SEED) -> List[dict]:
    title = "CH Count vs Node Count"
    out_file = os.path.join(out_dir, "exp2_ch_count_vs_node_count.csv")
    log_exp_start(2, title)

    rows: List[dict] = []

    for node_count in [30, 50, 100]:
        ba_m = 3
        ch_options = [2, 4, 6, 8, 10, 12]
        ch_options = [c for c in ch_options if 1 < c < node_count]

        # targeted remodification:
        # - source is not CH
        # - smaller deadline so delivery is meaningful
        # - harsher settings for small-node case
        source_node = node_count - 1
        max_rounds = max(6, math.ceil(math.log2(node_count)) * 2)

        if node_count <= 30:
            ch_overlay_degree = 1
            member_degree = 1
            member_forward_limit = 1
            transmission_success_prob = 0.95
        elif node_count <= 50:
            ch_overlay_degree = 2
            member_degree = 1
            member_forward_limit = 1
            transmission_success_prob = 0.95
        else:
            ch_overlay_degree = 2
            member_degree = 2
            member_forward_limit = 2
            transmission_success_prob = 0.95

        log(
            f"Exp 2 node_count={node_count} ongoing "
            f"(source={source_node}, max_rounds={max_rounds}, "
            f"overlay_deg={ch_overlay_degree}, member_deg={member_degree}, "
            f"member_fwd={member_forward_limit}, success_prob={transmission_success_prob})"
        )

        for ch_count in ch_options:
            run_results = []
            service_limits: List[float] = []

            for i in range(runs):
                seed = base_seed + i
                sim = BASimulator(seed)
                g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)

                res, used_limit = sim.run_cluster_fixed_exp2(
                    g=g,
                    source=source_node,
                    ch_count=ch_count,
                    ch_service_limit=None,
                    ch_overlay_degree=ch_overlay_degree,
                    member_degree=member_degree,
                    member_forward_limit=member_forward_limit,
                    transmission_success_prob=transmission_success_prob,
                    max_rounds=max_rounds,
                )
                run_results.append(res)
                service_limits.append(used_limit)

            row = summarize_results(
                2,
                {
                    "node_count": node_count,
                    "ch_count": ch_count,
                    "ba_m": ba_m,
                    "source_node": source_node,
                    "max_rounds_used": max_rounds,
                    "ch_overlay_degree": ch_overlay_degree,
                    "member_degree": member_degree,
                    "member_forward_limit": member_forward_limit,
                    "transmission_success_prob": transmission_success_prob,
                    "ch_service_limit_used": round(sum(service_limits) / len(service_limits), 4),
                },
                run_results,
            )
            rows.append(row)

        log(f"Exp 2 node_count={node_count} done")

    write_csv(out_file, rows)
    log_exp_done(2, title, out_file)
    return rows


def experiment_3_topology_density_vs_performance(out_dir: str, runs: int = RUNS, base_seed: int = BASE_SEED) -> List[dict]:
    title = "Topology Density vs Performance"
    out_file = os.path.join(out_dir, "exp3_topology_density_vs_performance.csv")
    log_exp_start(3, title)

    rows: List[dict] = []
    node_count = 100
    fanout = 3

    for ba_m in [1, 2, 3, 4, 6, 8]:
        run_results = []
        for i in range(runs):
            seed = base_seed + i
            sim = BASimulator(seed)
            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            res = sim.run_gossip(g=g, source=0, fanout=fanout)
            run_results.append(res)

        rows.append(
            summarize_results(3, {"node_count": node_count, "ba_m": ba_m, "fanout": fanout}, run_results)
        )

    write_csv(out_file, rows)
    log_exp_done(3, title, out_file)
    return rows


def experiment_4_ch_overload_failure(out_dir: str, runs: int = RUNS, base_seed: int = BASE_SEED) -> List[dict]:
    title = "CH Overload / Failure"
    out_file = os.path.join(out_dir, "exp4_ch_overload_failure.csv")
    log_exp_start(4, title)

    rows: List[dict] = []
    node_count = 120
    ch_count = 6
    ba_m = 3

    for overload_limit in [None, 20, 10, 5, 2]:
        run_results = []
        for i in range(runs):
            seed = base_seed + i
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
            seed = base_seed + i
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

    write_csv(out_file, rows)
    log_exp_done(4, title, out_file)
    return rows


def experiment_5_churn_sensitivity(out_dir: str, runs: int = RUNS, base_seed: int = BASE_SEED) -> List[dict]:
    title = "Churn Sensitivity"
    out_file = os.path.join(out_dir, "exp5_churn_sensitivity.csv")
    log_exp_start(5, title)

    rows: List[dict] = []
    node_count = 100
    ba_m = 3
    ch_count = 5
    fanout = 3

    for churn_rate in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40]:
        gossip_runs = []
        cluster_runs = []

        for i in range(runs):
            seed = base_seed + i
            sim = BASimulator(seed)

            g = sim.build_ba_graph(node_count=node_count, ba_m=ba_m)
            chs, node_to_ch = sim.derive_clusters_from_graph(g, ch_count=ch_count)

            all_nodes = list(g.nodes())
            remove_count = int(churn_rate * node_count)
            removed = set(sim.rng.sample(all_nodes[1:], min(remove_count, node_count - 1)))
            active = set(all_nodes) - removed

            gossip_runs.append(sim.run_gossip(g=g, source=0, fanout=fanout, active_nodes=active))

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

    write_csv(out_file, rows)
    log_exp_done(5, title, out_file)
    return rows


def experiment_6_heterogeneous_resources(out_dir: str, runs: int = RUNS, base_seed: int = BASE_SEED) -> List[dict]:
    title = "Heterogeneous Resources"
    out_file = os.path.join(out_dir, "exp6_heterogeneous_resources.csv")
    log_exp_start(6, title)

    rows: List[dict] = []
    node_count = 100
    ba_m = 3
    fanout = 4

    for scenario in ["homogeneous", "heterogeneous"]:
        run_results = []

        for i in range(runs):
            seed = base_seed + i
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

    write_csv(out_file, rows)
    log_exp_done(6, title, out_file)
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
    log("Starting experiment pipeline")
    out_dir = build_results_dirname()
    ensure_new_dir(out_dir)

    exp1 = experiment_1_fanout_vs_duplication(out_dir=out_dir, runs=RUNS, base_seed=BASE_SEED)
    exp2 = experiment_2_ch_count_vs_node_count(out_dir=out_dir, runs=RUNS, base_seed=BASE_SEED)
    exp3 = experiment_3_topology_density_vs_performance(out_dir=out_dir, runs=RUNS, base_seed=BASE_SEED)
    exp4 = experiment_4_ch_overload_failure(out_dir=out_dir, runs=RUNS, base_seed=BASE_SEED)
    exp5 = experiment_5_churn_sensitivity(out_dir=out_dir, runs=RUNS, base_seed=BASE_SEED)
    exp6 = experiment_6_heterogeneous_resources(out_dir=out_dir, runs=RUNS, base_seed=BASE_SEED)

    print_summary("Exp 1 - Fanout vs Duplication", exp1)
    print_summary("Exp 2 - CH Count vs Node Count", exp2)
    print_summary("Exp 3 - Topology Density vs Performance", exp3)
    print_summary("Exp 4 - CH Overload / Failure", exp4)
    print_summary("Exp 5 - Churn Sensitivity", exp5)
    print_summary("Exp 6 - Heterogeneous Resources", exp6)

    log(f"CSV files written to: {os.path.abspath(out_dir)}")
    log(f"RUNS = {RUNS}, BASE_SEED = {BASE_SEED}")
    log("Experiment pipeline completed successfully")


if __name__ == "__main__":
    main()