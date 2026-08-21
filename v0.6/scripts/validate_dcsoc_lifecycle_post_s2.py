"""Focused S2 lifecycle: replacement, local repair, return, recovery, du."""
from __future__ import annotations

import sys

import networkx as nx

from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


def fixture():
    graph = nx.Graph()
    graph.add_edges_from([(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3),
                          (3, 4), (4, 5),
                          (6, 7), (6, 8), (6, 9), (7, 8), (7, 9), (8, 9),
                          (5, 6)])
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=1.0, min_samples=4)
    sim = Simulator(nodes, DCSOCStrategy(3, 1), seed=42, base_delay=1.0, jitter=0.0,
                    cluster_manager=manager, controller=None, strategy_name="dcsoc")
    return sim


def main() -> int:
    sim = fixture()
    mgr = sim.cluster_manager
    failed = mgr.cluster_to_head[0]
    unaffected_before = {e for e in mgr.structural_edges if failed not in e}
    sim.nodes[failed].fail()
    replacement = sim.handle_dcsoc_failure(failed, was_core=True)
    unaffected_after = {e for e in mgr.structural_edges if failed not in e and replacement not in e}
    local_ok = unaffected_after.issubset(unaffected_before)

    sim.inject_message(replacement, "missed-while-away")
    sim.run(until=20.0)
    sim.handle_churn_join(sim.clock, failed, 0.1)
    sim.run(until=sim.clock + 5.0)
    return_ok = sim.nodes[failed].dcsoc_role == "leaf" and mgr.cluster_to_head[0] == replacement
    recovery_ok = (sim.nodes[failed].has_seen("missed-while-away") and
                   mgr.recovery_count == mgr.recovery_request_count == mgr.recovery_transfer_count == 1 and
                   mgr.recovery_time == 1.0)

    before_generation = mgr.structural_generation
    before_reclusters = mgr.recluster_count
    sim.schedule_event(sim.clock, 0, "dcsoc_recluster", {"eps": 1.0, "min_samples": 4})
    sim.run(until=sim.clock + 1.0)
    recluster_ok = (mgr.structural_generation == before_generation + 1 and
                    mgr.recluster_count == before_reclusters + 1)

    checks = {
        "core replaced": replacement is not None and replacement != failed and mgr.core_replacement_count == 1,
        "local repair counted": mgr.structural_repair_count == 1 and mgr.topology_edges_changed > 0,
        "unaffected relationships retained": local_ok,
        "former core returns as leaf": return_ok,
        "simulator-time recovery": recovery_ok,
        "explicit du regeneration": recluster_ok,
        "AHBN controller calls": sim.controller is None,
    }
    for label, ok in checks.items():
        print(f"{'PASS' if ok else 'FAIL'} — {label}")
    print(f"failed={failed} replacement={replacement} repair={mgr.structural_repair_count} "
          f"changed_edges={mgr.topology_edges_changed} recovery_time={mgr.recovery_time} "
          f"generation={mgr.structural_generation}")
    return 0 if all(checks.values()) else 1


if __name__ == "__main__":
    sys.exit(main())
