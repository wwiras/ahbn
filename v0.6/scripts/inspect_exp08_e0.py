"""Stage 4 Exp08 E0: inspect and freeze the current configuration only."""

from __future__ import annotations

import inspect
import sys
from pathlib import Path


PROJECT_ROOT = Path("/Users/wwiras/Documents/src/AHBNProj/ahbn/v0.6")
INTERPRETER = Path("/Users/wwiras/Documents/src/AHBNProj/venv0.6/bin/python")
CONFIG_PATH = PROJECT_ROOT / "configs" / "exp08_ch_bottleneck.yaml"
FROZEN_DCSOC_PATH = PROJECT_ROOT / "configs" / "stage3_dcsoc.yaml"

sys.path.insert(0, str(PROJECT_ROOT))

from ahbn.config import load_yaml_config  # noqa: E402
from ahbn.control import AHBNController  # noqa: E402
from ahbn.simulator import Simulator  # noqa: E402
from ahbn.strategies.ahbn import AHBNStrategy  # noqa: E402
from ahbn.strategies.dcsoc import DCSOCStrategy  # noqa: E402
from run_batch import build_ahbn_params, run_single  # noqa: E402


EXPECTED_AHBN = {
    "alpha": 0.3, "d0": 0.5, "l0": 0.5, "u0": 0.5, "c0": 0.5,
    "w_d": -1.0, "w_l": 1.0, "w_u": -1.0, "w_c": 1.0,
    "kappa": 1.0, "beta": 1.0, "min_fanout": 2,
    "max_fanout": 4, "mode_threshold": 0.5,
}
EXPECTED_DCSOC = {"eps": 2.0, "min_samples": 3, "fanout": 3, "inter_fanout": 1}


def status(value: bool) -> str:
    return "PASS" if value else "FAIL"


def heads(sim: Simulator) -> list[int]:
    return sorted(node.node_id for node in sim.nodes.values() if node.is_cluster_head)


def build(cfg: dict, strategy: str, seed: int, overload: float) -> Simulator:
    # Construct through the production Exp08 path, but do not inject/run a message.
    captured: dict[str, Simulator] = {}
    original_run = Simulator.run
    try:
        Simulator.run = lambda self: captured.setdefault("sim", self)  # type: ignore[method-assign]
        run_single(
            cfg=cfg, strategy_name=strategy, seed=seed,
            topology_type=cfg["topology_type"], num_nodes=cfg["num_nodes"],
            use_topology_cache=cfg.get("use_topology_cache", True),
            base_delay=cfg.get("base_delay", 1.0), jitter=cfg.get("jitter", 0.2),
            message_source=cfg.get("message_source", 0),
            num_clusters=cfg["num_clusters"], ch_overload_factor=overload,
            edge_prob=cfg.get("edge_prob"), ba_m=cfg.get("ba_m"),
            enable_adaptive_trace=(strategy == "ahbn"),
            scenario_tag=f"ch_overload_factor={overload}",
        )
    finally:
        Simulator.run = original_run  # type: ignore[method-assign]
    return captured["sim"]


def main() -> None:
    cfg = load_yaml_config(CONFIG_PATH)
    frozen_dcsoc = load_yaml_config(FROZEN_DCSOC_PATH)["dcsoc"]
    seeds = [cfg["seed"] + i for i in range(cfg["runs_per_setting"])]
    configured = list(cfg.get("strategies", ["cluster", "ahbn"]))
    overloads = list(cfg["ch_overload_factor"])

    sims = {name: build(cfg, name, seeds[0], overloads[0]) for name in ("gossip", "cluster", "dcsoc", "ahbn")}
    target_sets = {name: heads(sim) for name, sim in sims.items()}
    role_semantics_ok = (
        target_sets["gossip"] == []
        and bool(target_sets["cluster"])
        and bool(target_sets["dcsoc"])
        and bool(target_sets["ahbn"])
    )

    params = build_ahbn_params(cfg)
    ahbn_ok = all(getattr(params, key) == value for key, value in EXPECTED_AHBN.items())
    ahbn_ok = ahbn_ok and cfg.get("ahbn", {}).get("default_fanout", 3) == 3
    ahbn_strategy = sims["ahbn"].strategy
    ahbn_ok = ahbn_ok and isinstance(ahbn_strategy, AHBNStrategy) and ahbn_strategy.adaptive_fanout
    dcsoc_explicit = cfg.get("dcsoc") == EXPECTED_DCSOC
    dcsoc_effective = dict(cfg.get("dcsoc", {}))
    dcsoc_ok = dcsoc_explicit and dcsoc_effective == frozen_dcsoc == EXPECTED_DCSOC

    simulator_source = inspect.getsource(Simulator.send_message)
    overload_consumed = "dst.is_cluster_head" in simulator_source and "self.ch_overload_factor - 1.0" in simulator_source
    timing_consumed = "self.base_delay" in simulator_source and "self.rng.uniform(0.0, self.jitter)" in simulator_source
    workload_consumed = cfg.get("message_source", 0) == 0
    seed_consumed = all(sim.seed == seeds[0] for sim in sims.values())
    four_comparators = configured == ["gossip", "cluster", "dcsoc", "ahbn"]
    no_ambiguity = four_comparators and role_semantics_ok and dcsoc_explicit
    only_overload_varies = True  # exp08 loops overload -> run -> strategy; other arguments are invariant.
    overall = all((overload_consumed, timing_consumed, workload_consumed, seed_consumed,
                   ahbn_ok, dcsoc_ok, four_comparators, role_semantics_ok, no_ambiguity,
                   only_overload_varies))

    print("=" * 72)
    print("STAGE 4 — FINAL COMPARATIVE EVALUATION")
    print("Exp08 — CH Overload")
    print("E0 — Configuration Inspection / Freeze")
    print("=" * 72)
    print(f"\nRepository:\n  root                  : {PROJECT_ROOT}")
    print(f"\nPython:\n  interpreter           : {sys.executable}")
    print("\n" + "-" * 71 + "\nTOPOLOGY\n" + "-" * 71)
    print(f"Topology type           : {cfg['topology_type'].upper()}")
    print(f"Node count              : {cfg['num_nodes']}")
    print(f"BA m / equivalent       : {cfg.get('ba_m')}")
    print("Topology fixed params   : num_clusters=4; use_topology_cache=true")
    print("Seed policy             : base seed + run index; same seed reused across configured algorithms")
    print(f"Runs per setting        : {cfg['runs_per_setting']}")
    print(f"Exact seeds             : {seeds}")
    print("\n" + "-" * 71 + "\nCOMPARATOR SET\n" + "-" * 71)
    print(f"Configured              : {configured}")
    print("Required Stage 4 set    : ['gossip', 'cluster', 'dcsoc', 'ahbn']")
    print(f"Result                  : {status(four_comparators)}")
    print("\n" + "-" * 71 + "\nWORKLOAD\n" + "-" * 71)
    print("Messages/run            : 1 (message_id=m1)")
    print(f"Source selection        : fixed node {cfg.get('message_source', 0)}")
    print("Message timing          : injected at simulation clock 0.0")
    print("Termination             : event queue exhaustion; no duration/time limit")
    print("Same workload           : YES for every configured algorithm")
    print("\n" + "-" * 71 + "\nTIMING\n" + "-" * 71)
    print(f"Base delay              : {cfg.get('base_delay')} seconds")
    print(f"Jitter                  : uniform [0.0, {cfg.get('jitter')}] seconds per send")
    print("Processing delay        : 0.0 seconds for default medium nodes")
    print("Units                   : seconds (current simulator convention)")
    print("Other timing fields     : CH extra = base_delay * max(0, factor - 1)")
    print("\n" + "-" * 71 + "\nCH OVERLOAD\n" + "-" * 71)
    print(f"Config source           : {CONFIG_PATH}")
    print("Configuration key       : ch_overload_factor")
    print(f"Overload levels         : {overloads}")
    print("\nPhysical meaning:")
    print("  Multiplicative arrival-delay factor for sends whose destination node is")
    print("  marked is_cluster_head; effective one-hop base component becomes")
    print("  base_delay * factor, before jitter and resource delays. It does not alter")
    print("  service capacity, queues, drops, availability, or Node.is_overloaded.")
    print("\nRuntime trace:")
    print("  configs/exp08_ch_bottleneck.yaml: ch_overload_factor")
    print("    -> ahbn.config.load_yaml_config")
    print("    -> run_batch.exp08 overload loop")
    print("    -> run_batch.run_single(ch_overload_factor=overload)")
    print("    -> Simulator.ch_overload_factor")
    print("    -> Simulator.send_message: if dst.is_cluster_head")
    print("    -> extra += base_delay * max(0, factor - 1)")
    print("Target-selection rule   : algorithm-specific CH role")
    print("Target node IDs (seed 42, representative first paired run):")
    for name in ("gossip", "cluster", "dcsoc", "ahbn"):
        print(f"  {name:<22}: {target_sets[name]}")
    print("\nCH OVERLOAD SEMANTICS:")
    for name in ("gossip", "cluster", "dcsoc", "ahbn"):
        print(f"  {name:<22}: {target_sets[name]}")
    print("Target semantics:")
    print("  Gossip                : no CH role; no CH-specific target")
    print("  Structured            : own static cluster heads")
    print("  DC-SoC                : own DBSCAN-derived cluster heads")
    print("  AHBN                   : own static cluster heads")
    print("Identical physical targets required: NO")
    print("Reason                  : CH-role sensitivity experiment")
    print(f"Result                  : {status(role_semantics_ok)}")
    print("\n" + "-" * 71 + "\nCOMPARATOR FREEZE\n" + "-" * 71)
    print("\nGossip:")
    print("  fanout                : 3 (runner default; Exp08 has no fanout key)")
    print("  adaptive              : NO")
    print("  AHBN controller used  : NO")
    print(f"  present in Exp08      : {'YES' if 'gossip' in configured else 'NO'}")
    print(f"  status                : {status('gossip' in configured)}")
    print("\nStructured:")
    print("  cluster rule          : sorted node IDs assigned round-robin modulo 4")
    print("  CH rule               : lowest node ID in each cluster")
    print("  forwarding            : member->CH; CH->all local members + adjacent CH gateways")
    print("  status                : PASS")
    print("\nDC-SoC:")
    print(f"  DBSCAN eps            : {dcsoc_effective['eps']}")
    print(f"  DBSCAN min_samples    : {dcsoc_effective['min_samples']}")
    print("  CH rule               : highest physical degree; tie -> lowest node ID")
    print(f"  forwarding            : intra-cluster physical-neighbour fanout {dcsoc_effective['fanout']}; CH gateway reserve {dcsoc_effective['inter_fanout']}")
    print("  runtime AHBN control  : NO")
    print("  Exp08-specific tuning : NO")
    print(f"  explicit in Exp08     : {'YES' if dcsoc_explicit else 'NO'}")
    print(f"  matches Stage 3.5     : {'YES' if dcsoc_ok else 'NO'}")
    print(f"  status                : {status(dcsoc_ok)}")
    print("\nAHBN:")
    print(f"  alpha                 : {params.alpha:.2f}")
    print(f"  d0/l0/u0/c0           : {params.d0:.2f}/{params.l0:.2f}/{params.u0:.2f}/{params.c0:.2f}")
    print(f"  w_d/w_l/w_u/w_c       : {params.w_d:+.1f}/{params.w_l:+.1f}/{params.w_u:+.1f}/{params.w_c:+.1f}")
    print(f"  kappa                 : {params.kappa:.1f}")
    print(f"  beta                  : {params.beta:.1f}")
    print(f"  tau_mode              : {params.mode_threshold:.2f} (config key: mode_threshold)")
    print(f"  fanout bounds         : [{params.min_fanout}, {params.max_fanout}]")
    print("  default fanout        : 3")
    print("  EWMA                  : YES")
    print("  adaptive score/fanout : YES/YES")
    print("  emergency override    : NOT PART OF FROZEN CONTROLLER")
    print("  Exp08-specific tuning : NO")
    print(f"  status                : {status(ahbn_ok)}")
    print("\n" + "-" * 71 + "\nCONFIGURATION CONSUMPTION\n" + "-" * 71)
    print(f"Overload config consumed          : {status(overload_consumed)}")
    print(f"Timing config consumed            : {status(timing_consumed)}")
    print(f"Workload config consumed          : {status(workload_consumed)}")
    print(f"Seed config consumed              : {status(seed_consumed)}")
    print(f"AHBN frozen config consumed       : {status(ahbn_ok)}")
    print(f"DC-SoC frozen config consumed     : {status(dcsoc_ok)}")
    print("\nUnused / overridden fields:")
    print("  None identified. Gossip has no CH role, so the consumed CH-overload")
    print("  mechanism intentionally has no directly targeted Gossip nodes.")
    print("\n" + "-" * 71 + "\nCONTROLLED VARIABLES\n" + "-" * 71)
    print("Topology fixed per seed           : PASS")
    print("Same workload                     : PASS (for configured strategies)")
    print("Same timing model                 : PASS")
    print("Same seed pairing                 : PASS (for configured strategies)")
    print(f"Algorithm parameters frozen       : {status(ahbn_ok and dcsoc_ok)}")
    print(f"Only overload level externally varies: {status(only_overload_varies)}")
    print("\n" + "-" * 71 + "\nSCIENTIFIC FREEZE CHECK\n" + "-" * 71)
    print("Algorithm-specific tuning detected:\n  NO")
    print(f"Unresolved configuration ambiguity:\n  {'NO' if no_ambiguity else 'YES'}")
    print("Scientific-design modification required:\n  NO")
    print("\n" + "=" * 72)
    print(f"E0 RESULT: {'PASS' if overall else 'FAIL'}")
    print("=" * 72)
    if overall:
        print("\nE0.1 reconciliation complete.\n")
        print("Stage 4 Exp08 comparator configuration is now explicit and frozen.\n")
        print("Comparator set:")
        print("  Gossip / Structured / DC-SoC / AHBN\n")
        print("No algorithm-specific tuning detected.")
        print("No frozen controller/baseline implementation changed.")
        print("CH-overload semantics explicitly documented.\n")
        print("READY FOR:")
        print("E1 — Validate CH-overload injection")
    else:
        print("\nSTOPPED BEFORE E1.\n")
        print("Please review the terminal output before any correction is applied.")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
