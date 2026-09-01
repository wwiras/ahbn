"""Stage 3.4 S8: validate that DC-SoC forwarding is structurally determined."""

from copy import deepcopy

import networkx as nx

from ahbn.control import NodeControlState
from ahbn.message import Message
from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED, N, BA_M = 42, 30, 3
EPS, MIN_SAMPLES = 2.0, 3
FANOUT, INTER_FANOUT = 3, 1


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def structural_snapshot(nodes, source_id):
    node = nodes[source_id]
    return {
        "cluster_id": node.cluster_id,
        "is_cluster_head": node.is_cluster_head,
        "neighbors": tuple(node.neighbors),
        "gateway_neighbors": tuple(node.gateway_neighbors),
        "active": tuple(sorted((node_id, other.is_active) for node_id, other in nodes.items())),
        "clusters": tuple(sorted((node_id, other.cluster_id) for node_id, other in nodes.items())),
    }


def control_snapshot(control):
    return {
        "mode": control.mode,
        "fanout": control.fanout,
        "score": control.score,
        "weight": control.weight,
        "d_hat": control.d_hat,
        "l_hat": control.l_hat,
        "u_hat": control.u_hat,
        "c_hat": control.c_hat,
    }


def eligible_local_targets(node, nodes):
    return [
        neighbor_id
        for neighbor_id in node.neighbors
        if neighbor_id != node.node_id
        and neighbor_id in nodes
        and nodes[neighbor_id].is_active
        and nodes[neighbor_id].cluster_id == node.cluster_id
    ]


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S8 — Forwarding remains structurally determined")
    print("=" * 72)

    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=EPS, min_samples=MIN_SAMPLES)
    strategy = DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT)
    simulator = Simulator(
        nodes=nodes,
        strategy=strategy,
        seed=SEED,
        cluster_manager=manager,
        experiment_name="stage3_dcsoc_s8",
        strategy_name="dcsoc",
    )
    message = Message(message_id="dcsoc-s8-decision", source_id=0, created_at=0.0)

    cases = []
    for node_id, node in sorted(nodes.items()):
        candidates = eligible_local_targets(node, nodes)
        if node.cluster_id is not None and not node.is_cluster_head and len(candidates) > FANOUT:
            cases.append((len(candidates), -node_id, node_id))
    assert cases, "FAIL: no non-CH DC-SoC forwarding case has more candidates than fanout."
    source_id = max(cases)[2]
    source = nodes[source_id]
    local_candidates = eligible_local_targets(source, nodes)

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Topology nodes      : {graph.number_of_nodes()}")
    print(f"  Topology edges      : {graph.number_of_edges()}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nStatic implementation inspection:")
    print("  DC-SoC forwarding   : Simulator.handle_receive()")
    print("                        -> DCSOCStrategy.select_targets()")
    print("                        -> same-cluster active physical neighbours")
    print("                        -> CH gateway neighbours when source is a CH")
    print("                        -> fixed fanout/inter-fanout + simulator.rng sampling")
    print("                        -> Simulator.send_message()")
    print("  AHBN state consulted: NO (node.control is not read by select_targets())")

    rng_state = simulator.rng.getstate()
    structure_before = structural_snapshot(nodes, source_id)
    control_before = control_snapshot(source.control)
    targets_before = strategy.select_targets(source, message, simulator)

    assert targets_before, "FAIL: baseline DC-SoC forwarding selected no targets."
    assert len(targets_before) <= FANOUT, "FAIL: baseline selection exceeds fixed fanout."
    assert len(targets_before) == len(set(targets_before)), "FAIL: duplicate baseline targets."
    assert set(targets_before).issubset(local_candidates), (
        "FAIL: baseline target is not an eligible active same-cluster physical neighbour."
    )

    print("\nBaseline forwarding case:")
    print(f"  Source node         : {source_id}")
    print(f"  Cluster             : {source.cluster_id}")
    print(f"  Is cluster head     : {'YES' if source.is_cluster_head else 'NO'}")
    print(f"  Physical neighbours : {source.neighbors}")
    print(f"  Eligible neighbours : {local_candidates}")
    print(f"  Gateway neighbours  : {source.gateway_neighbors}")
    print(f"  Fixed fanout        : {strategy.fanout}")
    print(f"  Fixed inter-fanout  : {strategy.inter_fanout} (not exercised by non-CH source)")
    print(f"  RNG seed/state      : seed={SEED}; state saved before selection")
    print(f"  Forwarding targets  : {targets_before}")

    source.control = NodeControlState(
        d_hat=1.0,
        l_hat=1.0,
        u_hat=1.0,
        c_hat=1.0,
        score=-1000.0,
        weight=0.0,
        mode="cluster",
        fanout=4,
    )
    control_after = control_snapshot(source.control)
    structure_after_control = structural_snapshot(nodes, source_id)
    fixed_policy_unchanged = strategy.fanout == FANOUT and strategy.inter_fanout == INTER_FANOUT
    structure_unchanged = structure_after_control == structure_before
    control_changed = control_after != control_before
    simulator.rng.setstate(rng_state)
    targets_after_control = strategy.select_targets(source, message, simulator)
    targets_identical = targets_after_control == targets_before

    assert control_changed, "FAIL: AHBN NodeControlState did not change."
    assert structure_unchanged, "FAIL: AHBN mutation changed DC-SoC structural state."
    assert fixed_policy_unchanged, "FAIL: AHBN mutation changed fixed DC-SoC policy."
    assert targets_identical, "FAIL: AHBN NodeControlState changed DC-SoC forwarding targets."

    print("\nAHBN control-state invariance:")
    print(f"  AHBN state before   : {control_before}")
    print(f"  AHBN state after    : {control_after}")
    print(f"  Control state changed: {pass_fail(control_changed)}")
    print(f"  Structure unchanged : {pass_fail(structure_unchanged)}")
    print(f"  Fixed policy unchanged: {pass_fail(fixed_policy_unchanged)}")
    print("  RNG unchanged/reset : PASS")
    print(f"  Targets before      : {targets_before}")
    print(f"  Targets after       : {targets_after_control}")
    print(f"  Targets identical   : {pass_fail(targets_identical)}")
    print(f"  AHBN control-state invariance : {pass_fail(targets_identical)}")

    removed_neighbor = targets_before[0]
    original_neighbors = list(source.neighbors)
    original_reverse_neighbors = list(nodes[removed_neighbor].neighbors)
    assert removed_neighbor in source.neighbors, "FAIL: selected structural target is not a physical neighbour."
    assert source_id in nodes[removed_neighbor].neighbors, "FAIL: physical link is not symmetric before change."
    source.neighbors.remove(removed_neighbor)
    nodes[removed_neighbor].neighbors.remove(source_id)
    modified_neighbors = list(source.neighbors)
    assert removed_neighbor not in eligible_local_targets(source, nodes), (
        "FAIL: removed physical neighbour remains structurally eligible."
    )
    assert control_snapshot(source.control) == control_after, "FAIL: AHBN state changed during structural test."
    assert strategy.fanout == FANOUT and strategy.inter_fanout == INTER_FANOUT, (
        "FAIL: fixed DC-SoC forwarding policy changed during structural test."
    )
    simulator.rng.setstate(rng_state)
    targets_after_structure = strategy.select_targets(source, message, simulator)
    structural_effect = targets_after_structure != targets_before and removed_neighbor not in targets_after_structure

    # Restore the in-memory topology after exercising the production decision.
    source.neighbors = original_neighbors
    nodes[removed_neighbor].neighbors = original_reverse_neighbors

    assert structural_effect, "FAIL: valid physical-link removal did not alter forwarding targets."

    print("\nStructural sensitivity:")
    print("  Structural field    : symmetric physical-neighbour link membership")
    print(f"  Change applied      : remove link {source_id} <-> {removed_neighbor}")
    print(f"  Original value      : {original_neighbors}")
    print(f"  Modified value      : {modified_neighbors}")
    print(f"  Original targets    : {targets_before}")
    print(f"  Modified targets    : {targets_after_structure}")
    print(f"  Removed target absent: {pass_fail(removed_neighbor not in targets_after_structure)}")
    print(f"  Targets changed     : {pass_fail(structural_effect)}")
    print(f"  Structural sensitivity : {pass_fail(structural_effect)}")

    policy_strategy = DCSOCStrategy(fanout=1, inter_fanout=INTER_FANOUT)
    simulator.rng.setstate(rng_state)
    targets_fanout_one = policy_strategy.select_targets(source, message, simulator)
    simulator.rng.setstate(rng_state)
    targets_fanout_three = strategy.select_targets(source, message, simulator)
    policy_effect = len(targets_fanout_one) == 1 and len(targets_fanout_three) == FANOUT
    assert policy_effect, "FAIL: fixed fanout did not bound target count as expected."

    print("\nFixed-policy sensitivity:")
    print("  Fanout before       : 1")
    print(f"  Fanout after        : {FANOUT}")
    print(f"  Targets before      : {targets_fanout_one}")
    print(f"  Targets after       : {targets_fanout_three}")
    print(f"  Policy effect       : {pass_fail(policy_effect)}")

    print("\nRequired assertions:")
    print("  Valid deterministic case           : PASS")
    print("  Baseline targets structurally valid: PASS")
    print("  AHBN state independently changed   : PASS")
    print("  AHBN-state target invariance       : PASS")
    print("  Valid structural change applied    : PASS")
    print("  Structural target effect observed  : PASS")
    print("  AHBN controller/strategy required  : NO")

    print("\n" + "-" * 72)
    print("S8 RESULT: PASS")
    print("Forwarding remains structurally determined.")
    print("-" * 72)
    print("DC-SoC forwarding was invariant under irrelevant AHBN runtime state")
    print("changes with structure, fixed policy, and RNG state held constant.")
    print("A relevant structural change altered the forwarding decision.")


if __name__ == "__main__":
    main()
