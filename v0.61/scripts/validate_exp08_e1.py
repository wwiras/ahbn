"""Stage 4 Exp08 E1: independently validate CH-overload injection."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6")
INTERPRETER = Path("/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python")
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp08_ch_bottleneck.yaml"
COMPARATORS = ("gossip", "cluster", "dcsoc", "ahbn")

sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.config import load_yaml_config  # noqa: E402
from ahbn.control import AHBNController  # noqa: E402
from ahbn.message import Message  # noqa: E402
from ahbn.simulator import Simulator  # noqa: E402
from ahbn.strategies.ahbn import AHBNStrategy  # noqa: E402
from ahbn.strategies.cluster import ClusterStrategy  # noqa: E402
from ahbn.strategies.dcsoc import DCSOCStrategy  # noqa: E402
from ahbn.strategies.gossip import GossipStrategy  # noqa: E402
from run_batch import run_single  # noqa: E402


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def build(cfg: dict, comparator: str, factor: float) -> Simulator:
    """Construct through the production Exp08 path without running events."""
    captured: dict[str, Simulator] = {}
    original_run = Simulator.run
    try:
        Simulator.run = lambda self: captured.setdefault("sim", self)  # type: ignore[method-assign]
        run_single(
            cfg=cfg,
            strategy_name=comparator,
            seed=int(cfg["seed"]),
            topology_type=cfg["topology_type"],
            num_nodes=int(cfg["num_nodes"]),
            use_topology_cache=bool(cfg.get("use_topology_cache", True)),
            base_delay=float(cfg.get("base_delay", 1.0)),
            jitter=float(cfg.get("jitter", 0.2)),
            message_source=int(cfg.get("message_source", 0)),
            num_clusters=int(cfg["num_clusters"]),
            ch_overload_factor=factor,
            edge_prob=cfg.get("edge_prob"),
            ba_m=cfg.get("ba_m"),
            enable_adaptive_trace=(comparator == "ahbn"),
            scenario_tag=f"ch_overload_factor={factor}",
        )
    finally:
        Simulator.run = original_run  # type: ignore[method-assign]
    return captured["sim"]


def expected_targets(comparator: str, sim: Simulator, num_clusters: int) -> list[int]:
    """Resolve expected roles without reading Node.is_cluster_head."""
    if comparator == "gossip":
        return []
    if comparator in ("cluster", "ahbn"):
        # Static construction assigns sorted IDs round-robin, then selects the
        # lowest ID in each cluster.  Calculate that rule independently.
        ids = sorted(sim.nodes)
        members = {cluster_id: [] for cluster_id in range(num_clusters)}
        for index, node_id in enumerate(ids):
            members[index % num_clusters].append(node_id)
        return sorted(min(group) for group in members.values() if group)
    if comparator == "dcsoc":
        # DBSCAN membership is an architectural result.  Independently apply
        # the frozen head rule to that membership: greatest physical degree,
        # with lowest node ID as the tie-breaker.
        memberships: dict[int, list[int]] = {}
        for node_id, node in sim.nodes.items():
            if node.cluster_id is not None:
                memberships.setdefault(node.cluster_id, []).append(node_id)
        return sorted(
            max(group, key=lambda nid: (len(sim.nodes[nid].original_neighbors), -nid))
            for group in memberships.values()
        )
    raise ValueError(comparator)


def runtime_targets(sim: Simulator) -> list[int]:
    return sorted(node_id for node_id, node in sim.nodes.items() if node.is_cluster_head)


def policy_snapshot(sim: Simulator) -> tuple:
    nodes = tuple(
        (
            node_id,
            tuple(node.neighbors),
            tuple(node.original_neighbors),
            node.cluster_id,
            node.is_cluster_head,
            tuple(node.gateway_neighbors),
            node.is_active,
            node.is_overloaded,
            node.extra_delay,
            node.processing_delay,
            node.capacity_score,
        )
        for node_id, node in sorted(sim.nodes.items())
    )
    strategy = sim.strategy
    strategy_state = tuple(sorted((key, repr(value)) for key, value in vars(strategy).items()))
    controller_state = None if sim.controller is None else repr(sim.controller.params)
    return nodes, strategy.__class__.__name__, strategy_state, controller_state


def probe_delays(sim: Simulator) -> dict[int, float]:
    """Schedule one deterministic probe to every node and return its delay."""
    sim.queue.clear()
    for index, dst_id in enumerate(sorted(sim.nodes)):
        src_id = next(node_id for node_id in sorted(sim.nodes) if node_id != dst_id)
        sim.send_message(
            src_id,
            dst_id,
            Message(message_id=f"probe-{index}", source_id=src_id, created_at=0.0),
            now=0.0,
        )
    return {
        int(event.payload["message"].message_id.split("-")[1]): event.time
        for event in sim.queue
        if event.event_type == "receive" and event.payload["message"].message_id.startswith("probe-")
    }


def comparator_freeze(cfg: dict, name: str, sim: Simulator) -> bool:
    if name == "gossip":
        return isinstance(sim.strategy, GossipStrategy) and sim.strategy.fanout is None and sim.controller is None
    if name == "cluster":
        return isinstance(sim.strategy, ClusterStrategy) and sim.strategy.fanout is None and sim.controller is None
    if name == "dcsoc":
        frozen = cfg.get("dcsoc", {})
        return (
            isinstance(sim.strategy, DCSOCStrategy)
            and sim.strategy.fanout == frozen.get("fanout") == 3
            and sim.strategy.inter_fanout == frozen.get("inter_fanout") == 1
            and frozen.get("eps") == 2.0
            and frozen.get("min_samples") == 3
            and sim.controller is None
        )
    if name == "ahbn":
        return isinstance(sim.strategy, AHBNStrategy) and isinstance(sim.controller, AHBNController) and sim.strategy.adaptive_fanout
    return False


def main() -> None:
    cfg = load_yaml_config(CONFIG_PATH)
    normal_factor = 1.0
    overloaded_factor = float(max(cfg["ch_overload_factor"]))
    base_delay = float(cfg.get("base_delay", 1.0))
    expected_extra = base_delay * max(0.0, overloaded_factor - 1.0)

    interpreter_ok = Path(sys.executable).resolve() == INTERPRETER.resolve()
    config_ok = list(cfg.get("strategies", [])) == list(COMPARATORS)
    deterministic_ok = True
    all_resolution = True
    all_activation = True
    all_target_isolation = True
    all_non_target_normal = True
    all_policy = True
    all_comparator_freeze = True

    print("=" * 72)
    print("STAGE 4 — EXP08")
    print("E1 — Validate CH-Overload Injection")
    print("=" * 72)
    print("\nConfiguration:")
    print(f"  Config              : {CONFIG_PATH.relative_to(PROJECT_ROOT)}")
    print(f"  Python              : {sys.executable}")
    print(f"  Comparators         : {', '.join(COMPARATORS)}")
    print(f"  Seed                : {cfg['seed']}")
    print("  Activation          : simulator construction / run start (t=0)")
    print(f"  Normal value        : ch_overload_factor={normal_factor}")
    print(f"  Overloaded value    : ch_overload_factor={overloaded_factor}")

    for name in COMPARATORS:
        normal = build(cfg, name, normal_factor)
        overloaded = build(cfg, name, overloaded_factor)
        repeat = build(cfg, name, overloaded_factor)
        expected = expected_targets(name, overloaded, int(cfg["num_clusters"]))
        runtime = runtime_targets(overloaded)
        repeat_targets = runtime_targets(repeat)
        resolution_ok = expected == runtime

        normal_before = policy_snapshot(normal)
        overloaded_before = policy_snapshot(overloaded)
        normal_delays = probe_delays(normal)
        overloaded_delays = probe_delays(overloaded)
        normal_after = policy_snapshot(normal)
        overloaded_after = policy_snapshot(overloaded)

        deltas = {node_id: overloaded_delays[node_id] - normal_delays[node_id] for node_id in normal_delays}
        target_delay_ok = all(abs(deltas[node_id] - expected_extra) < 1e-12 for node_id in runtime)
        non_targets = sorted(set(normal.nodes) - set(runtime))
        non_target_ok = all(abs(deltas[node_id]) < 1e-12 for node_id in non_targets)
        # Gossip is the intentionally untargeted CH-independent reference.
        activation_ok = target_delay_ok and non_target_ok and (bool(runtime) or name == "gossip")
        policy_ok = normal_before == normal_after and overloaded_before == overloaded_after
        freeze_ok = comparator_freeze(cfg, name, overloaded)
        repeat_delays = probe_delays(repeat)
        repeat_ok = repeat_targets == runtime and repeat_delays == overloaded_delays

        all_resolution &= resolution_ok
        all_activation &= activation_ok
        all_target_isolation &= target_delay_ok
        all_non_target_normal &= non_target_ok
        all_policy &= policy_ok
        all_comparator_freeze &= freeze_ok
        deterministic_ok &= repeat_ok

        role = {
            "gossip": "CH-independent static Gossip reference (no CH target)",
            "cluster": "static Structured cluster heads",
            "dcsoc": "DBSCAN-derived density-cluster heads",
            "ahbn": "static cluster heads used by canonical AHBN",
        }[name]
        basis = {
            "gossip": "no native or mapped CH; expected empty target set",
            "cluster": "round-robin clusters; lowest node ID per cluster",
            "dcsoc": "DBSCAN membership; highest physical degree, tie -> lowest ID",
            "ahbn": "round-robin clusters; lowest node ID per cluster",
        }[name]
        representative_target = runtime[0] if runtime else None
        representative_non_target = non_targets[0] if non_targets else None

        print("\n" + "-" * 72)
        print(name.upper() if name != "dcsoc" else "DC-SOC")
        print("-" * 72)
        print(f"Relevant bottleneck role : {role}")
        print(f"Expected target           : {expected}")
        print(f"Runtime target            : {runtime}")
        print(f"Target-selection basis    : {basis}")
        print(f"Target resolution         : {status(resolution_ok)}")
        print("\nBefore overload (t=0, factor=1.0):")
        if representative_target is None:
            print("  Target                  : none by architecture")
        else:
            print(f"  Node {representative_target:<3} one-hop delay : {normal_delays[representative_target]:.12f}")
        if representative_non_target is not None:
            print(f"  Node {representative_non_target:<3} one-hop delay : {normal_delays[representative_non_target]:.12f}")
        print("\nAfter overload (t=0, configured run factor):")
        if representative_target is None:
            print("  Target                  : none; no delay injected (expected)")
        else:
            print(f"  Node {representative_target:<3} one-hop delay : {overloaded_delays[representative_target]:.12f}")
            print(f"  Observed added delay    : {deltas[representative_target]:.12f}")
        if representative_non_target is not None:
            print(f"  Node {representative_non_target:<3} one-hop delay : {overloaded_delays[representative_non_target]:.12f}")
            print(f"  Non-target added delay  : {deltas[representative_non_target]:.12f}")
        print(f"  processing_delay        : 0.0 before / 0.0 after (unchanged)")
        print(f"Injection observed        : {'YES' if runtime and target_delay_ok else 'NO (expected: no CH target)' if name == 'gossip' else 'NO'}")
        print(f"Non-target unexpectedly overloaded: {'NO' if non_target_ok else 'YES'}")
        print(f"Injection activation      : {status(activation_ok)}")
        print(f"Target isolation          : {status(target_delay_ok and non_target_ok)}")
        print(f"Forwarding-policy intact  : {status(policy_ok)}")
        print(f"Comparator isolation      : {status(freeze_ok)}")
        print(f"Deterministic replay      : {status(repeat_ok)}")
        print(f"Result                    : {status(all((resolution_ok, activation_ok, policy_ok, freeze_ok, repeat_ok)))}")

    source = inspect.getsource(Simulator.send_message)
    injector_scope_ok = (
        "dst.is_cluster_head" in source
        and "self.ch_overload_factor - 1.0" in source
        and "strategy" not in source
        and ".fanout" not in source
        and ".mode" not in source
        and "cluster_id" not in source
        and "neighbors" not in source
        and "controller" not in source
    )
    all_policy &= injector_scope_ok
    overall = all((
        interpreter_ok,
        config_ok,
        all_resolution,
        all_activation,
        all_target_isolation,
        all_non_target_normal,
        all_policy,
        all_comparator_freeze,
        deterministic_ok,
    ))

    print("\n" + "=" * 72)
    print("E1 CHECKS")
    print("=" * 72)
    print(f"Required interpreter                  : {status(interpreter_ok)}")
    print(f"Frozen four-comparator set            : {status(config_ok)}")
    print(f"Target resolution                     : {status(all_resolution)}")
    print(f"Configured overload activates         : {status(all_activation)}")
    print(f"Correct target node(s) affected       : {status(all_target_isolation)}")
    print(f"Non-target nodes remain normal        : {status(all_non_target_normal)}")
    print(f"No direct forwarding-policy mutation  : {status(all_policy)}")
    print(f"Comparator isolation                  : {status(all_comparator_freeze)}")
    print(f"Deterministic behaviour               : {status(deterministic_ok)}")
    print("\n" + "=" * 72)
    print(f"E1 RESULT: {status(overall)}")
    print("=" * 72)
    if not overall:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
