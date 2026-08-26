"""
Stage 3.4 — DC-SoC Sanity Validation
S3: Intra-cluster dissemination observed

Runs the production DC-SoC strategy and simulator, while observing the
transport events that the simulator actually schedules. The observer does
not select targets or create forwarding events.
"""

from collections import defaultdict

import networkx as nx

from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED = 42
N = 30
BA_M = 3
EPS = 2.0
MIN_SAMPLES = 3

# Frozen values from configs/stage3_dcsoc.yaml.
FANOUT = 3
INTER_FANOUT = 1
BASE_DELAY = 1.0
JITTER = 0.2
MESSAGE_SOURCE = 0


class ObservingSimulator(Simulator):
    """Record genuine point-to-point receive events without changing them."""

    def __init__(self, *args, **kwargs):
        self.forwarding_events = []
        super().__init__(*args, **kwargs)

    def schedule_event(self, time, priority, event_type, payload):
        # inject_message() schedules source -> source. It is deliberately not
        # forwarding evidence. All non-self receive events are created by the
        # production Simulator.send_message() transport path.
        if (
            event_type == "receive"
            and payload["src_id"] != payload["dst_id"]
        ):
            self.forwarding_events.append(
                (payload["src_id"], payload["dst_id"])
            )
        super().schedule_event(time, priority, event_type, payload)


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S3 — Intra-cluster dissemination observed")
    print("=" * 72)

    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    cluster_manager = assign_dcsoc_clusters(
        nodes,
        eps=EPS,
        min_samples=MIN_SAMPLES,
    )

    strategy = DCSOCStrategy(
        fanout=FANOUT,
        inter_fanout=INTER_FANOUT,
    )
    simulator = ObservingSimulator(
        nodes=nodes,
        strategy=strategy,
        seed=SEED,
        base_delay=BASE_DELAY,
        jitter=JITTER,
        cluster_manager=cluster_manager,
        experiment_name="stage3_dcsoc_s3",
        strategy_name="dcsoc",
    )

    memberships = defaultdict(list)
    for node_id, node in nodes.items():
        memberships[node.cluster_id].append(node_id)
    for members in memberships.values():
        members.sort()

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Topology nodes      : {graph.number_of_nodes()}")
    print(f"  Topology edges      : {graph.number_of_edges()}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nCluster summary:")
    for cluster_id in sorted(memberships):
        head_id = cluster_manager.cluster_to_head.get(cluster_id)
        print(f"  Cluster {cluster_id}:")
        print(f"    Members           : {memberships[cluster_id]}")
        print(f"    Cluster head      : {head_id}")

    source_cluster = nodes[MESSAGE_SOURCE].cluster_id
    source_is_ch = nodes[MESSAGE_SOURCE].is_cluster_head
    print("\nTransaction:")
    print(f"  Source node         : {MESSAGE_SOURCE}")
    print(f"  Source cluster      : {source_cluster}")
    print(f"  Source is CH        : {'YES' if source_is_ch else 'NO'}")

    simulator.inject_message(MESSAGE_SOURCE, "dcsoc-s3-transaction")
    simulator.run()

    observed = []
    for sender_id, receiver_id in simulator.forwarding_events:
        sender_cluster = nodes[sender_id].cluster_id
        receiver_cluster = nodes[receiver_id].cluster_id
        if sender_cluster == receiver_cluster and sender_cluster != -1:
            observed.append(
                (
                    sender_id,
                    receiver_id,
                    sender_cluster,
                    receiver_cluster,
                    nodes[sender_id].is_cluster_head,
                    nodes[receiver_id].is_cluster_head,
                )
            )

    print("\nObserved intra-cluster forwarding:")
    print("  sender -> receiver   sender_cluster  receiver_cluster  CH status")
    if observed:
        for sender, receiver, sender_cluster, receiver_cluster, sender_ch, receiver_ch in observed:
            ch_status = (
                f"sender={'YES' if sender_ch else 'NO'}, "
                f"receiver={'YES' if receiver_ch else 'NO'}"
            )
            print(
                f"  {sender:>6} -> {receiver:<8}"
                f"{sender_cluster:>8}{receiver_cluster:>18}  {ch_status}"
            )
    else:
        print("  (none)")

    actual_forwarding_ok = bool(simulator.forwarding_events)
    intra_cluster_ok = bool(observed)
    membership_ok = bool(observed) and all(
        sender in memberships[sender_cluster]
        and receiver in memberships[receiver_cluster]
        for sender, receiver, sender_cluster, receiver_cluster, _, _ in observed
    )
    non_noise_ok = (
        source_cluster is not None
        and source_cluster != -1
        and bool(observed)
        and all(event[2] != -1 for event in observed)
    )

    checks = [
        actual_forwarding_ok,
        intra_cluster_ok,
        membership_ok,
        non_noise_ok,
    ]

    print("\nValidation checks:")
    print(f"  Actual forwarding events observed            : {pass_fail(actual_forwarding_ok)}")
    print(f"  Intra-cluster forwarding observed            : {pass_fail(intra_cluster_ok)}")
    print(f"  Sender/receiver cluster membership valid     : {pass_fail(membership_ok)}")
    print(f"  Non-noise cluster used                       : {pass_fail(non_noise_ok)}")

    print("\n" + "=" * 72)
    if all(checks):
        print("S3 RESULT: PASS")
        print("=" * 72)
        return

    print("S3 RESULT: FAIL")
    if not actual_forwarding_ok:
        print("  - No actual non-self forwarding event was scheduled.")
    if not intra_cluster_ok:
        print("  - No forwarding event joined nodes in the same non-noise cluster.")
    if not membership_ok:
        print("  - Reported sender/receiver membership is inconsistent with Node state.")
    if not non_noise_ok:
        print("  - The source or observed forwarding did not use a non-noise cluster.")
    print("=" * 72)
    raise SystemExit(1)


if __name__ == "__main__":
    main()
