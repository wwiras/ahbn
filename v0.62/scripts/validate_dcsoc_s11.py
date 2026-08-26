"""Stage 3.4 S11: validate DC-SoC runtime isolation from AHBN."""

from dataclasses import asdict
from unittest.mock import patch

import networkx as nx

from ahbn.control import AHBNController
from ahbn.simulator import Simulator
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED, N, BA_M = 42, 30, 3
EPS, MIN_SAMPLES = 2.0, 3
FANOUT, INTER_FANOUT = 3, 1
BASE_DELAY, JITTER = 1.0, 0.2
SOURCE_ID, TRANSACTION_ID = 0, "1"


class TracingSimulator(Simulator):
    """Observe the production DC-SoC call path without changing it."""

    def __init__(self, *args, **kwargs):
        self.receive_calls = 0
        self.select_calls = 0
        self.send_calls = 0
        self.forwarding_events = []
        super().__init__(*args, **kwargs)

    def handle_receive(self, now, dst_id, src_id, message, sent_at=None):
        self.receive_calls += 1
        return super().handle_receive(now, dst_id, src_id, message, sent_at)

    def send_message(self, src_id, dst_id, message, now):
        queue_size = len(self.queue)
        result = super().send_message(src_id, dst_id, message, now)
        if len(self.queue) > queue_size:
            self.forwarding_events.append((src_id, dst_id))
        self.send_calls += 1
        return result


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def snapshot_ahbn_state(nodes):
    """Capture every field of every node's independent AHBN state object."""
    return {
        node_id: asdict(node.control)
        for node_id, node in sorted(nodes.items())
    }


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S11 — AHBN-controller isolation confirmed")
    print("=" * 72)

    graph = nx.barabasi_albert_graph(n=N, m=BA_M, seed=SEED)
    nodes = build_nodes_from_graph(graph)
    manager = assign_dcsoc_clusters(nodes, eps=EPS, min_samples=MIN_SAMPLES)
    strategy = DCSOCStrategy(fanout=FANOUT, inter_fanout=INTER_FANOUT)
    simulator = TracingSimulator(
        nodes=nodes,
        strategy=strategy,
        seed=SEED,
        base_delay=BASE_DELAY,
        jitter=JITTER,
        cluster_manager=manager,
        experiment_name="stage3_dcsoc_s11",
        strategy_name="dcsoc",
    )

    state_before = snapshot_ahbn_state(nodes)
    calls = {
        "simulator_update": 0,
        "controller_update": 0,
        "adaptive_decision": 0,
    }
    original_simulator_update = Simulator.update_ahbn_state
    original_controller_update = AHBNController.update_metrics
    original_adaptive_decision = AHBNController.decide_mode_and_fanout
    original_select_targets = DCSOCStrategy.select_targets

    def count_simulator_update(self, *args, **kwargs):
        calls["simulator_update"] += 1
        return original_simulator_update(self, *args, **kwargs)

    def count_controller_update(self, *args, **kwargs):
        calls["controller_update"] += 1
        return original_controller_update(self, *args, **kwargs)

    def count_adaptive_decision(self, *args, **kwargs):
        calls["adaptive_decision"] += 1
        return original_adaptive_decision(self, *args, **kwargs)

    def count_select_targets(self, *args, **kwargs):
        simulator.select_calls += 1
        return original_select_targets(self, *args, **kwargs)

    with (
        patch.object(Simulator, "update_ahbn_state", count_simulator_update),
        patch.object(AHBNController, "update_metrics", count_controller_update),
        patch.object(
            AHBNController,
            "decide_mode_and_fanout",
            count_adaptive_decision,
        ),
        patch.object(DCSOCStrategy, "select_targets", count_select_targets),
    ):
        simulator.inject_message(SOURCE_ID, TRANSACTION_ID)
        simulator.run()

    state_after = snapshot_ahbn_state(nodes)
    controller_ok = simulator.controller is None
    mutation_ok = state_before == state_after
    runtime_control_ok = (
        calls["controller_update"] == 0
        and calls["adaptive_decision"] == 0
    )
    forwarding_ok = (
        simulator.receive_calls > 0
        and simulator.select_calls > 0
        and simulator.send_calls > 0
        and bool(simulator.forwarding_events)
        and isinstance(simulator.strategy, DCSOCStrategy)
        and simulator.controller is None
        and runtime_control_ok
        and mutation_ok
    )

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Node count          : {N}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nController isolation:\n")
    print("Simulator.controller:")
    print(f"  {simulator.controller!r}")
    print("\nResult:")
    print(f"  {pass_fail(controller_ok)}")

    print("\nAHBN state mutation:\n")
    print("Before == After")
    print("\nMutation detected:")
    print(f"  {'NO' if mutation_ok else 'YES'}")
    print("\nResult:")
    print(f"  {pass_fail(mutation_ok)}")

    print("\nAHBN runtime control activity:\n")
    print("Simulator.update_ahbn_state():")
    print(f"  {calls['simulator_update']}")
    print("\nAHBNController.update_metrics():")
    print(f"  {calls['controller_update']}")
    print("\nAHBN adaptive decisions:")
    print(f"  {calls['adaptive_decision']}")
    print("\nResult:")
    print(f"  {pass_fail(runtime_control_ok)}")

    print("\nForwarding path:\n")
    print("Simulator.handle_receive()")
    print(" ->")
    print("DCSOCStrategy.select_targets()")
    print(" ->")
    print("physical neighbours / CH gateway logic")
    print(" ->")
    print("send_message()")
    print("\nObserved calls:")
    print(f"  handle_receive        : {simulator.receive_calls}")
    print(f"  select_targets        : {simulator.select_calls}")
    print(f"  send_message          : {simulator.send_calls}")
    print("\nAHBN intervention:")
    print(f"  {'NO' if forwarding_ok else 'YES'}")
    print("\nResult:")
    print(f"  {pass_fail(forwarding_ok)}")

    passed = controller_ok and mutation_ok and runtime_control_ok and forwarding_ok
    print("\n" + "=" * 72)
    print(f"S11 RESULT: {pass_fail(passed)}")
    print("=" * 72)
    assert passed, "S11 DC-SoC runtime isolation validation failed."


if __name__ == "__main__":
    main()
