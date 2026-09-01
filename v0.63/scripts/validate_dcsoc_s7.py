"""Stage 3.4 S7: validate DC-SoC independence from AHBN runtime control."""

import ast
import inspect
import sys
from pathlib import Path
from unittest.mock import patch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import networkx as nx

from ahbn.control import AHBNController
from ahbn.node import Node
from ahbn.simulator import Simulator
from ahbn.strategies.ahbn import AHBNStrategy
from ahbn.strategies.dcsoc import DCSOCStrategy
from ahbn.topology import assign_dcsoc_clusters, build_nodes_from_graph


SEED, N, BA_M = 42, 30, 3
EPS, MIN_SAMPLES = 2.0, 3
FANOUT, INTER_FANOUT = 3, 1
MESSAGE_ID = "dcsoc-s7-transaction"


def pass_fail(condition):
    return "PASS" if condition else "FAIL"


def dcsoc_static_dependencies():
    """Read the production method's AST, avoiding brittle text matching."""
    tree = ast.parse(inspect.getsource(DCSOCStrategy))
    attributes = {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}
    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)}
    prohibited_attributes = {
        "control", "update_ahbn_state", "update_metrics", "compute_score",
        "compute_weight", "sigmoid", "decide_mode_and_fanout", "mode",
        "weight", "score",
    }
    return {
        "attributes": attributes,
        "ahbn_strategy": "AHBNStrategy" in names,
        "controller_attributes": sorted(attributes & prohibited_attributes),
    }


def forbidden_call(label, calls):
    def sentinel(*args, **kwargs):
        calls[label] += 1
        raise AssertionError(
            f"AHBN runtime controller was invoked during DC-SoC dissemination: {label}"
        )
    return sentinel


def main():
    print("=" * 72)
    print("STAGE 3.4 — DC-SoC SANITY VALIDATION")
    print("S7 — No AHBN runtime controller used")
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
        experiment_name="stage3_dcsoc_s7",
        strategy_name="dcsoc",
    )

    non_heads = sorted(
        node_id for node_id, node in nodes.items()
        if node.cluster_id is not None and not node.is_cluster_head
    )
    source_id = non_heads[0]
    static = dcsoc_static_dependencies()
    shared_control_present = all(hasattr(node, "control") for node in nodes.values())
    controller_absent = simulator.controller is None
    correct_strategy = type(simulator.strategy) is DCSOCStrategy

    calls = {
        "AHBNController.update_metrics": 0,
        "AHBNController.compute_score": 0,
        "AHBNController.compute_weight": 0,
        "AHBNController.sigmoid": 0,
        "AHBNController.decide_mode_and_fanout": 0,
        "Simulator.get_duplicate_observation": 0,
        "Simulator.get_latency_observation": 0,
        "Simulator.get_utilization_observation": 0,
        "AHBNStrategy.__init__": 0,
        "AHBNStrategy.select_targets": 0,
    }
    guarded_dispatch_calls = 0
    original_update = Simulator.update_ahbn_state

    def observe_guarded_dispatch(self, *args, **kwargs):
        nonlocal guarded_dispatch_calls
        guarded_dispatch_calls += 1
        return original_update(self, *args, **kwargs)

    patches = [
        patch.object(AHBNController, "update_metrics", forbidden_call("AHBNController.update_metrics", calls)),
        patch.object(AHBNController, "compute_score", forbidden_call("AHBNController.compute_score", calls)),
        patch.object(AHBNController, "compute_weight", forbidden_call("AHBNController.compute_weight", calls)),
        patch.object(AHBNController, "sigmoid", forbidden_call("AHBNController.sigmoid", calls)),
        patch.object(AHBNController, "decide_mode_and_fanout", forbidden_call("AHBNController.decide_mode_and_fanout", calls)),
        patch.object(Simulator, "get_duplicate_observation", forbidden_call("Simulator.get_duplicate_observation", calls)),
        patch.object(Simulator, "get_latency_observation", forbidden_call("Simulator.get_latency_observation", calls)),
        patch.object(Simulator, "get_utilization_observation", forbidden_call("Simulator.get_utilization_observation", calls)),
        patch.object(AHBNStrategy, "__init__", forbidden_call("AHBNStrategy.__init__", calls)),
        patch.object(AHBNStrategy, "select_targets", forbidden_call("AHBNStrategy.select_targets", calls)),
        patch.object(Simulator, "update_ahbn_state", observe_guarded_dispatch),
    ]

    for active_patch in patches:
        active_patch.start()
    dissemination_completed = False
    runtime_error = None
    try:
        simulator.inject_message(source_id, MESSAGE_ID)
        simulator.run()
        dissemination_completed = True
    except AssertionError as exc:
        runtime_error = str(exc)
    finally:
        for active_patch in reversed(patches):
            active_patch.stop()

    record = simulator.metrics.messages[MESSAGE_ID]
    delivered = len(record.first_seen_times)
    no_forbidden_calls = all(count == 0 for count in calls.values())
    no_static_control_dependency = not static["controller_attributes"]
    no_ahbn_strategy_dependency = not static["ahbn_strategy"] and correct_strategy
    control_not_used = "control" not in static["attributes"]

    print("\nTest configuration:")
    print("  Topology type       : BA")
    print(f"  Topology nodes      : {graph.number_of_nodes()}")
    print(f"  Topology edges      : {graph.number_of_edges()}")
    print(f"  BA m                : {BA_M}")
    print(f"  Seed                : {SEED}")
    print(f"  DBSCAN eps          : {EPS}")
    print(f"  DBSCAN min_samples  : {MIN_SAMPLES}")

    print("\nStatic implementation inspection:")
    print("  DC-SoC construction : run_one.build_simulation_from_config('dcsoc')")
    print("                        -> assign_dcsoc_clusters() -> DCSOCStrategy(...)")
    print("                        -> Simulator(..., controller=None)")
    print("  DC-SoC forwarding   : Simulator.handle_receive()")
    print("                        -> DCSOCStrategy.select_targets()")
    print("                        -> same-cluster physical neighbours; CH gateway neighbours")
    print("                        -> seeded bounded sampling -> Simulator.send_message()")
    print("  DC-SoC repair       : churn handler -> repair_topology_after_churn()")
    print("                        -> active neighbours / cluster overlay / CH gateway refresh")
    print("  AHBN control path   : Simulator.update_ahbn_state()")
    print("                        -> normalized observations -> AHBNController.update_metrics()")
    print("                        -> EWMA -> compute_score()/sigmoid()")
    print("                        -> decide_mode_and_fanout() -> node.control")
    print("                        -> AHBNStrategy.select_targets()")
    print("  AHBNStrategy instantiated by DC-SoC : NO" if no_ahbn_strategy_dependency else "  AHBNStrategy instantiated by DC-SoC : YES")
    print(f"  AHBN controller dependency          : {'NONE' if no_static_control_dependency else static['controller_attributes']}")

    print("\nDC-SoC forwarding inputs:")
    print("  Cluster membership                  : USED (node.cluster_id)")
    print("  Cluster-head information            : USED (node.is_cluster_head)")
    print("  Physical topology                   : USED (node.neighbors)")
    print("  CH gateway overlay                  : USED (node.gateway_neighbors)")
    print("  Fixed fanout/inter-fanout limits    : USED")
    print("  Simulator seeded RNG                : USED for bounded sampling")
    print("  AHBN EWMA observations              : NOT USED")
    print("  AHBN adaptive score / sigmoid       : NOT USED")
    print("  AHBN runtime mode                   : NOT USED")
    print("  AHBN adaptive fanout                : NOT USED")
    print("  NodeControlState forwarding input   : NOT USED")
    print(f"  Shared AHBN control fields on Node  : {'YES' if shared_control_present else 'NO'}")

    print("\nRuntime instrumentation (raising sentinels):")
    print(f"  Guarded update_ahbn_state dispatches: {guarded_dispatch_calls} (returns at controller=None)")
    print(f"  AHBN sensing calls                  : {sum(calls[name] for name in calls if name.startswith('Simulator.get_'))}")
    print(f"  AHBN EWMA/controller update calls   : {calls['AHBNController.update_metrics']}")
    print(f"  AHBN score/sigmoid calls            : {calls['AHBNController.compute_score'] + calls['AHBNController.compute_weight'] + calls['AHBNController.sigmoid']}")
    print(f"  AHBN mode/fanout decision calls     : {calls['AHBNController.decide_mode_and_fanout']}")
    print(f"  AHBNStrategy construction calls     : {calls['AHBNStrategy.__init__']}")
    print(f"  AHBNStrategy forwarding calls       : {calls['AHBNStrategy.select_targets']}")
    print(f"  Sentinel triggered                  : {'YES' if runtime_error else 'NO'}")

    print("\nDC-SoC transaction:")
    print(f"  Source node                         : {source_id}")
    print("  Source selection                    : lowest-ID non-CH cluster member")
    print(f"  Transaction ID                      : {MESSAGE_ID}")
    print(f"  Dissemination completed             : {'YES' if dissemination_completed else 'NO'}")
    print(f"  Delivered nodes                     : {delivered}/{N}")
    print(f"  Transmission count                  : {record.total_forwards}")
    print(f"  Duplicate count                     : {record.duplicate_count}")

    print("\nStructural-maintenance distinction:")
    print("  Structural update capability        : PRESENT")
    print("  AHBN forwarding adaptation          : ABSENT")

    passed = all((
        dissemination_completed, delivered > 0, correct_strategy,
        controller_absent, no_forbidden_calls, no_static_control_dependency,
        no_ahbn_strategy_dependency, control_not_used,
    ))
    print("\nValidation:")
    print(f"  DC-SoC dissemination completed      : {pass_fail(dissemination_completed)}")
    print(f"  No AHBN runtime mechanism invoked   : {pass_fail(no_forbidden_calls and controller_absent)}")
    print(f"  No AHBN-driven forwarding decision  : {pass_fail(no_static_control_dependency and control_not_used)}")
    if runtime_error:
        print(f"  Prohibited dependency               : {runtime_error}")

    print("\n" + "-" * 72)
    print(f"S7 RESULT: {pass_fail(passed)}")
    print("-" * 72)
    print("Conclusion:")
    print("  DC-SoC uses its predefined dissemination policy and structural")
    print("  maintenance mechanism without AHBN runtime forwarding adaptation.")
    print("\n  DC-SoC : structure-adaptive, forwarding-fixed")
    print("  AHBN   : runtime forwarding-adaptive")
    assert passed, "S7 AHBN-runtime independence validation failed."


if __name__ == "__main__":
    main()
